from dataclasses import dataclass
from typing import Optional, List, Dict, Set

# from vllm.logger import init_logger
from deft.memory_pool import ReqToTokenPool, TokenToKVPool
from deft.tree_decoding.tree_index_pool import TreeIndexPool
from transformers import AutoTokenizer

import torch
import math

# logger = init_logger(__name__)

from .timer import GlobalTimer


class UnpagedKVCache:
    def __init__(self, layer_num: int) -> None:
        self.data: List[Optional[torch.Tensor]] = [
            None for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        data = self.data[layer_id]
        assert data is not None
        return data[:, 0]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        data = self.data[layer_id]
        assert data is not None
        return data[:, 1]

    def merge(self, other: "UnpagedKVCache") -> None:
        for idx, data in enumerate(other.data):
            assert data is not None
            layer_data = self.data[idx]
            if layer_data is None:
                self.data[idx] = data
            else:
                self.data[idx] = torch.cat((layer_data, data), 0)

    def update(self, layer_id: int, new_data: torch.Tensor) -> None:
        assert len(new_data.shape) == 4  # (s, 2, head_num, head_size)
        layer_data = self.data[layer_id]
        if layer_data is None:
            self.data[layer_id] = new_data
        else:
            assert new_data.shape[1:] == layer_data.shape[1:]
            self.data[layer_id] = torch.cat((layer_data, new_data), 0)


class KVCacheUpdater:
    def __init__(
        self,
        use_paged_memory: bool,
        token_to_kv_pool: Optional[TokenToKVPool],
        cache_loc: Optional[torch.Tensor],
        leaf_data: Optional[list[UnpagedKVCache]],
        is_prompt: bool,
    ):
        self.use_paged_memory = use_paged_memory
        self.token_to_kv_pool = token_to_kv_pool
        self.cache_loc = cache_loc
        self.unpaged_cache = leaf_data
        self.is_prompt = is_prompt

    def update(
        self, layer_id: int, cache_k: torch.Tensor, cache_v: torch.Tensor
    ) -> None:
        if self.use_paged_memory:
            assert self.token_to_kv_pool is not None
            assert self.cache_loc is not None
            key_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
            value_buffer = self.token_to_kv_pool.get_value_buffer(layer_id)
            key_buffer[self.cache_loc] = cache_k
            value_buffer[self.cache_loc] = cache_v
        else:
            assert self.token_to_kv_pool is None
            assert self.cache_loc is None
            assert self.unpaged_cache is not None
            if self.is_prompt:
                new_data = torch.cat(
                    (cache_k.unsqueeze(1), cache_v.unsqueeze(1)), dim=1
                )
                self.unpaged_cache[0].update(layer_id, new_data)
            else:
                for idx, cache in enumerate(self.unpaged_cache):
                    new_data = torch.stack(
                        (cache_k[idx], cache_v[idx])
                    ).unsqueeze(0)
                    cache.update(layer_id, new_data)


class TreeNode:
    def __init__(
        self,
        id: int,
        node_indices_id: Optional[int] = None,
        node_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.id = id
        self.children: Dict[int, TreeNode] = {}
        self.token_ids: List[int] = []
        self.positions: List[int] = []
        self.position_offset = 0
        self.kv_indices: List[int] = []
        self.kv_data: Optional[UnpagedKVCache] = None
        self.parent: Optional[TreeNode] = None
        self.refs: Set[TreeNode] = set()
        self.paused = False
        self.node_indices_id = node_indices_id
        self.node_indices = node_indices
        # NOTE(jinwei): hard code to support beam search based on cumulative logprob.
        self.cumulative_logprob = 0.0

    def get_len(self) -> int:
        return len(self.token_ids)

    def append_token(self, token: int, logprob: Optional[float] = None) -> None:
        self.positions.append(self.position_offset + len(self.token_ids))
        self.token_ids.append(token)
        if logprob is not None:
            self.cumulative_logprob += logprob

    def append_index(self, index: int) -> None:
        self.kv_indices.append(index)
        if self.node_indices is not None:
            node_len = len(self.kv_indices)
            self.node_indices[node_len - 1] = index


class BranchSequence:
    # inspired by class Sequence in vLLM. We design class BranchSequence to record the finished branches of tree decoding.
    def __init__(self, id: int):
        self.id = id
        self.token_ids: List[int] = []
        self.cumulative_logprob = 0.0
        self.PPL = 0.0

    def get_len(self) -> int:
        return len(self.token_ids)

    def append_tokens(self, tokens: List[int]) -> None:
        self.token_ids.extend(tokens)


class TreeCache:
    def __init__(
        self,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool: Optional[TokenToKVPool],
        tree_index_pool: Optional[TreeIndexPool],
        use_paged_memory: bool = True,
        use_tree_index: bool = False,
    ) -> None:
        self.node_cnt = 1
        self.root: Optional[TreeNode] = None
        self.nodes: dict[int, TreeNode] = {}
        self.leaves: Dict[int, TreeNode] = {}
        self.leaf_to_req: Dict[int, int] = {}
        # self.path_tokens = {}
        self.paused_nodes: Set[int] = set()  # record the paused nodes.
        self.leaf_to_q: Dict[int, int] = {}
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.tree_index_pool = tree_index_pool
        self.use_paged_memory = use_paged_memory
        self.use_tree_index = use_tree_index
        self.layer_num = layer_num
        self.deleted_token_num = 0
        if self.use_paged_memory:
            assert token_to_kv_pool is not None
            assert req_to_token_pool is not None
        else:
            assert token_to_kv_pool is None
            assert req_to_token_pool is None
        if self.use_tree_index:
            assert self.use_paged_memory
            assert self.tree_index_pool is not None

        # TODO(jinwei): design a class of sampler to enable user to decide the branch criteria easily. Refer to vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sampler.py
        self.all_finished_seqs: List[
            BranchSequence
        ] = []  # Use class BranchSequence  to record the results of finished branches which include seq len, tokens, cum_probs.

        # self.template_tree= None # as a backbone of execution for the tree decoding

    def init_prompt(self, prompt_ids: torch.Tensor) -> KVCacheUpdater:
        self.root = TreeNode(0)
        self.nodes[0] = self.root
        self.root.token_ids = prompt_ids.tolist()
        self.root.position_offset = 0
        self.root.positions = list(range(len(prompt_ids)))
        self.leaves[self.root.id] = self.root
        self.add_ref(self.root)

        if self.use_paged_memory:
            assert self.req_to_token_pool is not None
            req = self.req_to_token_pool.alloc(1)
            assert req is not None
            req_id = int(req[0].item())
            self.leaf_to_req[self.root.id] = req_id
            assert self.token_to_kv_pool is not None
            cache_loc = self.token_to_kv_pool.alloc(len(prompt_ids))
            assert cache_loc is not None
            self.root.kv_indices = cache_loc.tolist()
            self.req_to_token_pool.req_to_token[req, : len(prompt_ids)] = (
                cache_loc
            )
            if self.use_tree_index:
                assert self.tree_index_pool is not None
                node_indices_ids = self.tree_index_pool.alloc(1)
                assert node_indices_ids is not None
                node_indices_id = int(node_indices_ids[0].item())
                self.root.node_indices = self.tree_index_pool.node_to_kv[
                    node_indices_id
                ]
                self.root.node_indices_id = node_indices_id
                self.root.node_indices[: len(prompt_ids)] = cache_loc
            return KVCacheUpdater(
                use_paged_memory=self.use_paged_memory,
                token_to_kv_pool=self.token_to_kv_pool,
                cache_loc=cache_loc,
                leaf_data=None,
                is_prompt=True,
            )
        else:
            self.root.kv_data = UnpagedKVCache(layer_num=self.layer_num)
            cache_list = [self.root.kv_data]
            return KVCacheUpdater(
                use_paged_memory=self.use_paged_memory,
                token_to_kv_pool=None,
                cache_loc=None,
                leaf_data=cache_list,
                is_prompt=True,
            )

    def new_node(self, parent: TreeNode) -> TreeNode:
        node_indices = None
        node_indices_id = None
        if self.use_tree_index:
            assert self.tree_index_pool is not None
            node_indices_ids = self.tree_index_pool.alloc(1)
            assert node_indices_ids is not None
            node_indices_id = int(node_indices_ids[0].item())
            node_indices = self.tree_index_pool.node_to_kv[node_indices_id]
        node = TreeNode(self.node_cnt, node_indices_id, node_indices)
        self.node_cnt += 1
        node.parent = parent
        node.position_offset = parent.position_offset + len(parent.positions)
        parent.children[node.id] = node
        self.nodes[node.id] = node
        if not self.use_paged_memory:
            node.kv_data = UnpagedKVCache(layer_num=self.layer_num)
        return node

    def alloc(self) -> KVCacheUpdater:
        if self.use_paged_memory:
            assert self.token_to_kv_pool is not None
            assert self.req_to_token_pool is not None
            out_cache_loc = self.token_to_kv_pool.alloc(len(self.leaves))
            assert out_cache_loc is not None
            for idx, leaf in enumerate(
                sorted(self.leaves.values(), key=lambda x: x.id)
            ):
                loc = int(out_cache_loc[idx].item())
                leaf.append_index(loc)
                req = self.leaf_to_req[leaf.id]
                self.req_to_token_pool.req_to_token[req, leaf.positions[-1]] = (
                    loc
                )

            return KVCacheUpdater(
                use_paged_memory=self.use_paged_memory,
                token_to_kv_pool=self.token_to_kv_pool,
                cache_loc=out_cache_loc,
                leaf_data=None,
                is_prompt=False,
            )
        else:
            leaf_list = list(sorted(self.leaves.values(), key=lambda x: x.id))
            cache_list: list[UnpagedKVCache] = []
            for leaf in leaf_list:
                cache = leaf.kv_data
                assert cache is not None
                cache_list.append(cache)
            return KVCacheUpdater(
                use_paged_memory=self.use_paged_memory,
                token_to_kv_pool=None,
                cache_loc=None,
                leaf_data=cache_list,
                is_prompt=False,
            )

    # merge kv cache and tokens of two nodes
    def merge_nodes(
        self,
        node_A: TreeNode,
        node_B: TreeNode,
        pruneB_flag: Optional[bool] = True,
    ) -> None:
        # merge the kv cache and tokens from B to A
        for token_id in node_B.token_ids:
            node_A.positions.append(
                node_A.position_offset + len(node_A.token_ids)
            )
            node_A.append_token(token=token_id)
        if self.use_paged_memory:
            for kv_idx in node_B.kv_indices:
                node_A.append_index(index=kv_idx)
            # debug:
            # # add refs of token_to_kv_pool;
            assert self.token_to_kv_pool is not None
            self.token_to_kv_pool.add_refs(torch.tensor(node_B.kv_indices))
        else:
            assert node_A.kv_data is not None and node_B.kv_data is not None
            node_A.kv_data.merge(node_B.kv_data)

        # after merge, remember to prune node B.
        if pruneB_flag:
            self.cut(node_B)

    def reset_node_KV(self, node: TreeNode, diff: int) -> None:
        if self.use_paged_memory:
            # # debug(jinwei): free token_to_KV_pool? Does not work even we free kv_indices
            assert self.token_to_kv_pool is not None
            self.token_to_kv_pool.free(torch.tensor(node.kv_indices))
            node.kv_indices = []
        else:  # to test
            node.kv_data = UnpagedKVCache(layer_num=self.layer_num)
        node.position_offset += diff
        node.positions = [pos + diff for pos in node.positions]

    def branch(self, node: TreeNode, branch_cnt: int) -> List[TreeNode]:
        # assert branch_cnt > 1
        assert node.id in self.leaves  # Comment to fit speculativde decoding
        self.leaves.pop(node.id)
        # if node.id in self.leaves:
        #     self.leaves.pop(node.id)
        req = 0
        path_len = node.positions[-1] + 1
        if self.use_paged_memory:
            req = self.leaf_to_req.pop(node.id)
        is_first = True
        new_nodes: List[TreeNode] = []
        for _ in range(branch_cnt):
            child = self.new_node(node)
            new_nodes.append(child)
            self.leaves[child.id] = child
            if self.use_paged_memory:
                assert self.req_to_token_pool is not None
                if is_first:
                    self.leaf_to_req[child.id] = req
                    is_first = False
                else:
                    new_req = self.req_to_token_pool.alloc(1)
                    assert new_req is not None
                    new_req_id = int(new_req[0].item())
                    self.req_to_token_pool.copy(req, new_req_id, path_len)
                    self.leaf_to_req[child.id] = new_req_id

        self.remove_ref(node)
        for child in new_nodes:
            self.add_ref(child)

        return new_nodes

    # TODO(jinwei): we should consider the case when the node we want to prune is not the leaf nodes.
    def cut(
        self, node: TreeNode, record_deleted: bool = False
    ) -> List[TreeNode]:
        assert len(node.children) == 0
        assert node.id in self.leaves
        self.leaves.pop(node.id)
        self.remove_ref(node)
        if self.use_paged_memory:
            req = self.leaf_to_req.pop(node.id)
            assert self.req_to_token_pool is not None
            self.req_to_token_pool.free(req)
        assert len(node.refs) == 0
        deleted_nodes = []
        cur: Optional[TreeNode] = node
        while cur is not None and len(cur.refs) == 0:
            deleted_nodes.append(self.nodes.pop(cur.id))
            if self.use_paged_memory:
                assert self.token_to_kv_pool is not None
                self.token_to_kv_pool.free(torch.tensor(cur.kv_indices))
                if self.use_tree_index:
                    assert self.tree_index_pool is not None
                    assert cur.node_indices_id is not None
                    self.tree_index_pool.free(cur.node_indices_id)
            parent = cur.parent
            if parent is not None:
                parent.children.pop(cur.id)
            cur = parent
        if record_deleted:
            for deleted in deleted_nodes:
                self.deleted_token_num += len(deleted.token_ids)
        return deleted_nodes

    # def pause(self, node: TreeNode):
    #     assert node.id in self.leaves
    #     assert node.paused == False
    #     node.paused = True
    #     self.paused_nodes.add(node.id)

    # def resume(self, node: TreeNode):
    #     assert node.id in self.leaves
    #     assert node.paused == True
    #     node.paused = False
    #     self.paused_nodes.remove(node.id)

    def get_kv_seq(self, layer_id: int) -> torch.Tensor:
        assert not self.use_paged_memory
        data: List[torch.Tensor] = []
        for leaf in sorted(self.leaves.values(), key=lambda x: x.id):
            cur_node: Optional[TreeNode] = leaf
            seq_data = None
            while cur_node is not None:
                kv_data = cur_node.kv_data
                assert kv_data is not None
                layer_data = kv_data.data[layer_id]
                assert layer_data is not None
                if seq_data is None:
                    seq_data = layer_data
                else:
                    seq_data = torch.cat((layer_data, seq_data), 0)
                cur_node = cur_node.parent
            assert seq_data is not None
            data.append(seq_data)
        data_tensor = torch.stack(data, 0)
        assert (
            len(data_tensor.shape) == 5
        )  # (batch_size, seq_len, 2, head_num, head_size)
        return data_tensor

    def get_kv_tree_with_mask(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.use_paged_memory
        data = None
        mask = None
        q_len = len(self.leaves)
        leaf_to_q = {
            leaf.id: idx
            for idx, leaf in enumerate(
                sorted(self.leaves.values(), key=lambda x: x.id)
            )
        }

        for node in sorted(self.nodes.values(), key=lambda x: x.id):
            kv_data = node.kv_data
            assert kv_data is not None
            layer_data = kv_data.data[
                layer_id
            ]  # (seq_len, 2, head_num, head_size)
            assert layer_data is not None
            seq_len, _, head_num, _ = layer_data.shape
            assert layer_data is not None
            seq_len = layer_data.shape[0]
            cur_mask = torch.full(
                (q_len,), float("-inf"), dtype=torch.float16, device="cuda"
            )
            for leaf in node.refs:
                idx = leaf_to_q[leaf.id]
                cur_mask[idx] = 0
            cur_mask = cur_mask.reshape(1, q_len, 1).expand(1, q_len, seq_len)
            if data is None:
                data = layer_data
                mask = cur_mask
            else:
                assert mask is not None
                data = torch.cat((data, layer_data), 0)
                mask = torch.cat((mask, cur_mask), dim=2)

        assert data is not None
        assert mask is not None
        return data, mask

    def get_kv_tree(self, layer_id: int) -> torch.Tensor:
        assert not self.use_paged_memory
        data = None

        for node in sorted(self.nodes.values(), key=lambda x: x.id):
            kv_data = node.kv_data
            assert kv_data is not None
            layer_data = kv_data.data[
                layer_id
            ]  # (seq_len, 2, head_num, head_size)
            assert layer_data is not None

            if data is None:
                data = layer_data
            else:
                data = torch.cat((data, layer_data), 0)

        assert data is not None
        return data

    def add_ref(self, node: TreeNode) -> None:
        ref = node
        node.refs.add(ref)
        while node.parent is not None:
            node = node.parent
            node.refs.add(ref)

    def remove_ref(self, node: TreeNode) -> None:
        ref = node
        node.refs.remove(ref)
        while node.parent is not None:
            node = node.parent
            node.refs.remove(ref)

    def free(self) -> None:  # free/evict the whole tree
        self.root = None
        self.nodes.clear()
        self.leaves.clear()
        # reset cnt
        self.node_cnt = 0

    def output_branch(self, dstnode: TreeNode) -> None:
        # path from root->dstnode
        path = self._find_path_to_node(dstnode)

        # BranchSequence
        branch_seq = BranchSequence(len(self.all_finished_seqs))

        # append tokens in the path to BranchSequence
        for node in path:
            branch_seq.append_tokens(node.token_ids)
            branch_seq.cumulative_logprob += node.cumulative_logprob

        branch_seq.PPL = math.exp(
            -branch_seq.cumulative_logprob / len(branch_seq.token_ids)
        )
        # add BranchSequence to all_finished_seqs
        self.all_finished_seqs.append(branch_seq)

    def _find_path_to_node(self, dstnode: TreeNode) -> List[TreeNode]:
        path = []
        node = dstnode
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def print_finished_branches(self, tokenizer: AutoTokenizer) -> None:
        print(
            f"Total number of generated branches={len(self.all_finished_seqs)}! \n"
        )
        for branch in self.all_finished_seqs:
            generated_text = tokenizer.decode(
                branch.token_ids, skip_special_tokens=True
            )
            print(
                f" Branch ID: {branch.id}\n",
                f"Generated Text: {generated_text}\n",
                f"Tokens in this path:{branch.token_ids}\n",
                f"Token length : {len(branch.token_ids)}\n",
                # f"Cumulative logprob: {branch.cumulative_logprob}\n"
                f"Perplexity: {branch.PPL}\n",
            )

    def get_tree_token_number(self) -> int:
        """
        Returns the total number of tokens present in the tree.

        This function sums up the number of tokens stored in each node
        (including leaves and intermediate nodes) of the tree.

        Returns:
            int: Total number of tokens in the tree.
        """
        total_tokens = 0
        for node in self.nodes.values():
            total_tokens += len(node.token_ids)  # Count tokens in each node

        total_tokens += self.deleted_token_num
        return total_tokens


BLOCK_CONFIG = {"BLOCK_LEN": 128, "MAX_BLOCK_LEN": -1}
TRAVERSAL_CONFIG = {"METHOD": "dfs"}


@dataclass
class TreeMetadata:
    query_num: int
    node_num: int
    total_kv_len: int
    leaf_to_q: Dict[int, int]
    node_q: torch.Tensor
    node_kv: torch.Tensor
    node_q_len: torch.Tensor
    node_kv_len: torch.Tensor
    node_q_offset: torch.Tensor
    node_kv_offset: torch.Tensor

    block_len: int

    block_q: torch.Tensor
    block_q_cnts: torch.Tensor
    block_q_offset: torch.Tensor

    # block_node_cnts: torch.Tensor
    block_bitmasks: torch.Tensor
    # block_node_lens: torch.Tensor
    # block_node_offset: torch.Tensor

    block_kv: torch.Tensor
    block_lens: torch.Tensor

    @classmethod
    def from_tree_cache(
        cls,
        tree: TreeCache,
        tile_num: int = 8,
        max_q_len: int = 32,
        max_block_len: int = -1,
    ) -> "TreeMetadata":
        node_q = []
        node_kv = []
        node_q_len = []
        node_kv_len = []
        block_q = []
        block_q_cnts = []
        block_bitmasks = []
        block_kv = []
        block_lens = []
        # block_node_cnts = []
        # block_node_lens = []
        leaf_to_q: Dict[int, int] = {}
        leaf_cnt = 0
        total_kv_len = 0

        # for node in tree.nodes.values():
        #     total_kv_len += len(node.kv_indices)

        # block_len = max(32, (total_kv_len + tile_num - 1) // tile_num )
        # print("block_len = ", block_len)
        block_len = BLOCK_CONFIG["BLOCK_LEN"]
        if max_block_len == -1:
            max_block_len = BLOCK_CONFIG["MAX_BLOCK_LEN"]

        for leaf in sorted(tree.leaves.values(), key=lambda x: x.id):
            leaf_to_q[leaf.id] = leaf_cnt
            leaf_cnt += 1
        cur_q_set_union: Set[int] = set()
        cur_q_sets: List[Set[int]] = []
        cur_block_kv: List[int] = []
        cur_block_node_cnt = 0
        cur_block_node_lens = []

        max_q_set_len = 0

        def pack_new_block() -> None:
            nonlocal cur_block_node_cnt
            nonlocal max_q_set_len
            # block_kv.extend(cur_block_kv)
            # block_lens.append(len(cur_block_kv))
            # block_node_cnts.append(cur_block_node_cnt)
            # block_node_lens.extend(cur_block_node_lens)
            cur_block_len = len(cur_block_kv)
            if cur_block_len < block_len:
                cur_block_kv.extend(
                    [-1 for _ in range(block_len - cur_block_len)]
                )
                cur_block_node_lens.append(block_len - cur_block_len)
                cur_q_sets.append(set())

            cur_q_set_sorted = sorted(cur_q_set_union)
            max_q_set_len = max(max_q_set_len, len(cur_q_set_union))

            processed_len = 0
            while processed_len < len(cur_q_set_sorted):
                partial_len = min(
                    max_q_len, len(cur_q_set_sorted) - processed_len
                )
                partial_q_set = cur_q_set_sorted[
                    processed_len : processed_len + partial_len
                ]
                partial_q_to_idx = {
                    q: idx for idx, q in enumerate(partial_q_set)
                }
                partial_q_bitmasks = [
                    sum(
                        [
                            1 << partial_q_to_idx[q]
                            for q in q_set
                            if q in partial_q_set
                        ]
                    )
                    for q_set in cur_q_sets
                ]
                block_q.extend(partial_q_set)
                block_q_cnts.append(len(partial_q_set))
                block_kv.extend(cur_block_kv)
                block_lens.append(cur_block_len)
                for bitmask, node_len in zip(
                    partial_q_bitmasks, cur_block_node_lens
                ):
                    block_bitmasks.extend([bitmask] * node_len)
                processed_len += partial_len

            # q_to_idx = {q: idx for idx, q in enumerate(sorted(cur_q_set_union))}
            # q_bitmasks = [sum([1 << q_to_idx[q] for q in q_set]) for q_set in cur_q_sets]
            # block_q.extend(sorted(cur_q_set_union))
            # block_q_cnts.append(len(cur_q_set_union))
            # max_q_set_len = max(max_q_set_len, len(cur_q_set_union))
            # # block_bitmasks.extend(q_bitmasks)
            # for bitmask, node_len in zip(q_bitmasks, cur_block_node_lens):
            #     block_bitmasks.extend([bitmask] * node_len)

            cur_q_set_union.clear()
            cur_q_sets.clear()
            cur_block_kv.clear()
            cur_block_node_cnt = 0
            cur_block_node_lens.clear()

        def dfs(node: TreeNode) -> None:
            assert len(node.refs) > 0
            assert len(node.token_ids) > 0

            if node.paused:
                return

            nonlocal cur_block_node_cnt
            nonlocal total_kv_len
            # nonlocal cur_block_node_lens
            if tree.use_paged_memory:
                kv = sorted(node.kv_indices)
            else:
                kv = [-1 for _ in range(len(node.token_ids))]

            total_kv_len += len(kv)
            q = sorted(
                [leaf_to_q[ref.id] for ref in node.refs if not ref.paused]
            )
            partial_kvs = []
            node_block = max_block_len
            if node_block == -1:
                node_block = len(kv)
            for i in range(0, len(kv), node_block):
                partial_kvs.append(kv[i : min(i + node_block, len(kv))])
            # q_set = set(q)
            # q_set_bitmask = sum([1 << x for x in q_set])
            for i in range(0, len(q), max_q_len):
                partial_q = q[i : min(i + max_q_len, len(q))]
                for partial_kv in partial_kvs:
                    node_q.extend(partial_q)
                    node_q_len.append(len(partial_q))
                    node_kv.extend(partial_kv)
                    node_kv_len.append(len(partial_kv))
            # node_q.extend(q)
            # node_q_len.append(len(q))
            # node_kv.extend(kv)
            # node_kv_len.append(len(kv))
            residual_len = block_len - len(cur_block_kv)
            packed_len = 0

            while packed_len < len(kv):
                if len(kv) - packed_len < residual_len:
                    partial_kv = kv[packed_len:]

                    cur_block_kv.extend(partial_kv)
                    cur_q_set_union.update(q)
                    cur_q_sets.append(set(q))
                    cur_block_node_cnt += 1
                    cur_block_node_lens.append(len(partial_kv))
                    break
                else:
                    partial_kv = kv[packed_len : packed_len + residual_len]

                    cur_block_kv.extend(partial_kv)
                    cur_q_set_union.update(q)
                    cur_q_sets.append(set(q))
                    cur_block_node_cnt += 1
                    cur_block_node_lens.append(len(partial_kv))

                    pack_new_block()

                    packed_len += residual_len
                    residual_len = block_len

            for child in node.children.values():
                dfs(child)

        GlobalTimer.start("traversal")
        assert tree.root is not None
        dfs(tree.root)

        if cur_block_node_cnt > 0:
            pack_new_block()
        GlobalTimer.stop("traversal")
        GlobalTimer.start("alloc")
        # for node in tree.nodes.values():
        #     assert len(node.refs) > 0
        #     if len(node.kv_indices) == 0:
        #         continue
        #     q = sorted([leaf_to_q[ref.id] for ref in node.refs])
        #     node_q.extend(q)
        #     node_q_len.append(len(q))

        #     kv = sorted(node.kv_indices)
        #     node_kv.extend(kv)
        #     node_kv_len.append(len(kv))

        node_q_tensor = torch.tensor(node_q, dtype=torch.long, device="cuda")
        node_kv_tensor = torch.tensor(node_kv, dtype=torch.long, device="cuda")
        node_q_len_tensor = torch.tensor(
            node_q_len, dtype=torch.long, device="cuda"
        )
        node_kv_len_tensor = torch.tensor(
            node_kv_len, dtype=torch.long, device="cuda"
        )
        node_q_offset_tensor = torch.cat(
            [
                torch.tensor([0], device="cuda"),
                torch.cumsum(node_q_len_tensor, dim=0)[:-1],
            ]
        )
        node_kv_offset_tensor = torch.cat(
            [
                torch.tensor([0], device="cuda"),
                torch.cumsum(node_kv_len_tensor, dim=0)[:-1],
            ]
        )

        block_q_tensor = torch.tensor(block_q, dtype=torch.long, device="cuda")
        block_q_cnts_tensor = torch.tensor(
            block_q_cnts, dtype=torch.long, device="cuda"
        )
        block_q_offset_tensor = torch.cat(
            [
                torch.tensor([0], device="cuda"),
                torch.cumsum(block_q_cnts_tensor, dim=0)[:-1],
            ]
        )

        block_bitmasks_tensor = torch.tensor(
            block_bitmasks, dtype=torch.long, device="cuda"
        )
        # block_node_cnts_tensor = torch.tensor(block_node_cnts, dtype=torch.long, device="cuda")
        # block_node_lens_tensor = torch.tensor(block_node_lens, dtype=torch.long, device="cuda")
        # block_node_offset_tensor = torch.cat([torch.tensor([0], device="cuda"), torch.cumsum(block_node_cnts_tensor, dim=0)[:-1]])

        block_lens_tensor = torch.tensor(
            block_lens, dtype=torch.long, device="cuda"
        )
        block_kv_tensor = torch.tensor(
            block_kv, dtype=torch.long, device="cuda"
        )
        GlobalTimer.stop("alloc")
        # print(f"kv len: {len(block_kv)}, q len: {len(block_q)}, max q set len: {max_q_set_len}")
        return cls(
            query_num=leaf_cnt,
            node_num=len(node_q_len),
            total_kv_len=total_kv_len,
            leaf_to_q=leaf_to_q,
            node_q=node_q_tensor,
            node_kv=node_kv_tensor,
            node_q_len=node_q_len_tensor,
            node_kv_len=node_kv_len_tensor,
            node_q_offset=node_q_offset_tensor,
            node_kv_offset=node_kv_offset_tensor,
            block_len=block_len,
            block_q=block_q_tensor,
            block_q_cnts=block_q_cnts_tensor,
            block_q_offset=block_q_offset_tensor,
            block_bitmasks=block_bitmasks_tensor,
            # block_node_cnts=block_node_cnts_tensor,
            # block_node_lens=block_node_lens_tensor,
            # block_node_offset=block_node_offset_tensor,
            block_lens=block_lens_tensor,
            block_kv=block_kv_tensor,
        )

    @classmethod
    def from_tree_cache_node(
        cls,
        tree: TreeCache,
        tile_num: int = 8,
        max_q_len: int = 32,
        max_block_len: int = -1,
    ) -> "TreeMetadata":
        assert tree.use_tree_index
        assert tree.tree_index_pool is not None
        node_q: list[int] = []
        node_kv_offset: list[int] = []
        node_q_len: list[int] = []
        node_kv_len: list[int] = []

        # block_node_cnts = []
        # block_node_lens = []
        leaf_to_q: Dict[int, int] = {}
        leaf_cnt = 0
        total_kv_len = 0

        # for node in tree.nodes.values():
        #     total_kv_len += len(node.kv_indices)

        # block_len = max(32, (total_kv_len + tile_num - 1) // tile_num )
        # print("block_len = ", block_len)
        block_len = BLOCK_CONFIG["BLOCK_LEN"]
        if max_block_len == -1:
            max_block_len = BLOCK_CONFIG["MAX_BLOCK_LEN"]

        for leaf in sorted(tree.leaves.values(), key=lambda x: x.id):
            leaf_to_q[leaf.id] = leaf_cnt
            leaf_cnt += 1

        def dfs(node: TreeNode) -> None:
            assert len(node.refs) > 0
            assert len(node.token_ids) > 0

            if node.paused:
                return

            nonlocal total_kv_len
            # nonlocal cur_block_node_lens
            assert node.node_indices is not None
            assert node.node_indices_id is not None
            assert tree.tree_index_pool is not None
            offset = tree.tree_index_pool.get_offset(node.node_indices_id)
            kv_len = len(node.kv_indices)

            total_kv_len += kv_len
            q = sorted(
                [leaf_to_q[ref.id] for ref in node.refs if not ref.paused]
            )
            partial_kv_offsets = []
            partial_kv_lens = []
            node_block = max_block_len
            if node_block == -1:
                node_block = kv_len
            for i in range(0, kv_len, node_block):
                partial_kv_offsets.append(i + offset)
                partial_kv_lens.append(min(node_block, kv_len - i))
            for i in range(0, len(q), max_q_len):
                partial_q = q[i : min(i + max_q_len, len(q))]
                for partial_kv_offset, partial_kv_len in zip(
                    partial_kv_offsets, partial_kv_lens
                ):
                    node_q.extend(partial_q)
                    node_q_len.append(len(partial_q))
                    node_kv_offset.append(partial_kv_offset)
                    node_kv_len.append(partial_kv_len)

            for child in node.children.values():
                dfs(child)

        GlobalTimer.start("traversal")
        assert tree.root is not None
        dfs(tree.root)
        GlobalTimer.stop("traversal")
        GlobalTimer.start("alloc")
        # for node in tree.nodes.values():
        #     assert len(node.refs) > 0
        #     if len(node.kv_indices) == 0:
        #         continue
        #     q = sorted([leaf_to_q[ref.id] for ref in node.refs])
        #     node_q.extend(q)
        #     node_q_len.append(len(q))

        #     kv = sorted(node.kv_indices)
        #     node_kv.extend(kv)
        #     node_kv_len.append(len(kv))

        node_q_tensor = torch.tensor(node_q, dtype=torch.long, device="cuda")
        node_kv_tensor = tree.tree_index_pool.node_to_kv.view(-1)
        node_q_len_tensor = torch.tensor(
            node_q_len, dtype=torch.long, device="cuda"
        )
        node_kv_len_tensor = torch.tensor(
            node_kv_len, dtype=torch.long, device="cuda"
        )
        node_q_offset_tensor = torch.cat(
            [
                torch.tensor([0], device="cuda"),
                torch.cumsum(node_q_len_tensor, dim=0)[:-1],
            ]
        )
        node_kv_offset_tensor = torch.tensor(
            node_kv_offset, dtype=torch.long, device="cuda"
        )

        null = torch.empty(0, dtype=torch.long, device="cuda")
        block_q_tensor = null
        block_q_cnts_tensor = null
        block_q_offset_tensor = null
        block_bitmasks_tensor = null
        block_lens_tensor = null
        block_kv_tensor = null
        GlobalTimer.stop("alloc")
        return cls(
            query_num=leaf_cnt,
            node_num=len(node_q_len),
            total_kv_len=total_kv_len,
            leaf_to_q=leaf_to_q,
            node_q=node_q_tensor,
            node_kv=node_kv_tensor,
            node_q_len=node_q_len_tensor,
            node_kv_len=node_kv_len_tensor,
            node_q_offset=node_q_offset_tensor,
            node_kv_offset=node_kv_offset_tensor,
            block_len=block_len,
            block_q=block_q_tensor,
            block_q_cnts=block_q_cnts_tensor,
            block_q_offset=block_q_offset_tensor,
            block_bitmasks=block_bitmasks_tensor,
            block_lens=block_lens_tensor,
            block_kv=block_kv_tensor,
        )


GLOBAL_TREE_METADATA: Optional[TreeMetadata] = None
GLOBAL_TREE_CACHE: Optional[TreeCache] = None


def register_tree_metadata(tree_metadata: TreeMetadata) -> None:
    global GLOBAL_TREE_METADATA
    GLOBAL_TREE_METADATA = tree_metadata


def unregister_tree_metadata() -> None:
    global GLOBAL_TREE_METADATA
    GLOBAL_TREE_METADATA = None


def get_global_tree_metadata() -> TreeMetadata:
    assert GLOBAL_TREE_METADATA is not None
    return GLOBAL_TREE_METADATA


def register_tree_cache(tree_cache: TreeCache) -> None:
    global GLOBAL_TREE_CACHE
    GLOBAL_TREE_CACHE = tree_cache


def unregister_tree_cache() -> None:
    global GLOBAL_TREE_CACHE
    GLOBAL_TREE_CACHE = None


def get_global_tree_cache() -> TreeCache:
    assert GLOBAL_TREE_CACHE is not None
    return GLOBAL_TREE_CACHE
