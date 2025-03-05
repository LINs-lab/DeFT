import torch
from deft.layers.attention.context_flashattention_nopad import (
    context_attention_fwd,
)
from deft.layers.attention.token_attention import token_attention_fwd
from deft.model_runner import ForwardMode, InputMetadata
from deft.layers.attention.tree_attention import (
    tree_attention_fwd,
    tree_attention_subtree_fwd,
)

# from deft.layers.Atten_Operator.unpaged.Flash_decoding import Flash_decoding_cuda
# from deft.layers.attention.unpaged.Flash_attention import Flash_attention_triton
from deft.layers.attention.unpaged.causal_masked import (
    tree_attention_causal_masked,
)
from deft.layers.attention.unpaged.tree_attention import (
    tree_attention_fwd as deft_node_unpaged,
)
from deft.layers.attention.unpaged.tree_attention import (
    tree_attention_subtree_fwd as deft_flatten_unpaged,
)
from torch import nn

from deft.tree_decoding.timer import GlobalTimer
from deft.tree_decoding.perf_metrics import PerfMetrics
from deft.tree_decoding.tree_cache import (
    get_global_tree_metadata,
    get_global_tree_cache,
)


class DeFTAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.scaling = scaling
        self.head_dim = head_dim
        self.layer_id = layer_id

    def prefill_forward_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        o = torch.empty_like(q)

        context_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
        )
        self.store_kv_cache(k, v, input_metadata)

        return o

    def deft_node_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        o = torch.zeros_like(q)
        self.store_kv_cache(k, v, input_metadata)
        tree_metadata = get_global_tree_metadata()
        assert tree_metadata is not None
        assert input_metadata.token_to_kv_pool is not None
        GlobalTimer.stop("attn_mem")
        PerfMetrics.update_KV_IO(
            int(tree_metadata.node_kv_len.sum().item()),
            self.tp_q_head_num * self.head_dim,
        )
        GlobalTimer.start("attn_comp")
        torch.cuda.nvtx.range_push("attn_comp")
        tree_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            tree_metadata.node_kv,
            tree_metadata.node_kv_offset,
            tree_metadata.node_kv_len,
            tree_metadata.node_q,
            tree_metadata.node_q_offset,
            tree_metadata.node_q_len,
        )
        GlobalTimer.stop("attn_comp")
        torch.cuda.nvtx.range_pop()
        return o

    def deft_flatten_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        o = torch.zeros_like(q)
        self.store_kv_cache(k, v, input_metadata)
        tree_metadata = get_global_tree_metadata()

        assert tree_metadata is not None
        assert input_metadata.token_to_kv_pool is not None
        GlobalTimer.stop("attn_mem")
        # PerfMetrics.update_KV_IO(int(tree_metadata.block_lens.sum().item()), self.tp_q_head_num * self.head_dim)
        # PerfMetrics.update_Mask_IO(int(tree_metadata.block_lens.sum().item()))
        PerfMetrics.update_KV_IO(
            tree_metadata.total_kv_len,
            self.tp_q_head_num * self.head_dim,
        )
        PerfMetrics.update_Mask_IO(tree_metadata.total_kv_len)
        torch.cuda.nvtx.range_push("attn_comp")
        GlobalTimer.start("attn_comp")
        tree_attention_subtree_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            tree_metadata.block_len,
            tree_metadata.block_q,
            tree_metadata.block_q_cnts,
            tree_metadata.block_q_offset,
            tree_metadata.block_bitmasks,
            tree_metadata.block_kv,
            tree_metadata.block_lens,
        )
        GlobalTimer.stop("attn_comp")
        torch.cuda.nvtx.range_pop()
        return o

    def radix_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")

        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        o = torch.zeros_like(q)
        self.store_kv_cache(k, v, input_metadata)
        assert input_metadata.token_to_kv_pool is not None
        assert input_metadata.req_to_token_pool is not None
        PerfMetrics.update_KV_IO(
            int(input_metadata.total_num_tokens),
            self.tp_q_head_num * self.head_dim,
        )
        GlobalTimer.stop("attn_mem")
        GlobalTimer.start("attn_comp")
        token_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
            input_metadata.other_kv_index,
            input_metadata.total_num_tokens,
        )
        GlobalTimer.stop("attn_comp")
        return o

    def medusa_unpaged_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        q = q.view(-1, self.tp_q_head_num, self.head_dim)
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        self.store_kv_cache(k, v, input_metadata)
        tree = get_global_tree_cache()
        kv, mask = tree.get_kv_tree_with_mask(self.layer_id)
        k, v = kv[:, 0], kv[:, 1]  # (seq_len, head_num, head_dim)
        PerfMetrics.update_Causal_Tree_Attn_IO(
            q.shape[0], k.shape[0], q.shape[1], q.shape[2]
        )
        group_size = self.tp_q_head_num // self.tp_k_head_num
        q = q.transpose(0, 1)
        k = (
            k.transpose(0, 1).transpose(1, 2).repeat_interleave(group_size, 0)
        )  # (head_num, head_dim, seq_len)
        v = v.transpose(0, 1).repeat_interleave(
            group_size, 0
        )  # (head_num, seq_len, head_dim)
        scale = 1.0 / (self.head_dim**0.5)
        GlobalTimer.stop("attn_mem")
        GlobalTimer.start("attn_comp")

        # print(k.shape, v.shape)
        # attn = torch.softmax(torch.matmul(q, k) * scale + mask, dim=2) # (head_num, batch_size, seq_len)
        # o = torch.matmul(attn, v) # (head_num, batch_size, head_dim)
        o = tree_attention_causal_masked(q, k, v, scale, mask)

        o = o.transpose(0, 1).reshape(-1, self.tp_q_head_num * self.head_dim)
        GlobalTimer.stop("attn_comp")
        return o

    def flash_decoding_unpaged_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        self.store_kv_cache(k, v, input_metadata)
        # group_size = self.tp_q_head_num // self.tp_k_head_num
        tree = get_global_tree_cache()
        kv = tree.get_kv_seq(self.layer_id)
        k, v = kv[:, :, 0], kv[:, :, 1]
        PerfMetrics.update_KV_IO(
            kv.shape[1] * kv.shape[0], self.tp_q_head_num * self.head_dim
        )
        GlobalTimer.stop("attn_mem")
        GlobalTimer.start("attn_comp")
        # scale = 1.0 / (self.head_dim ** 0.5)

        # o = x(
        #     q.view(-1, 1, self.tp_q_head_num, self.head_dim),
        #     k,
        #     v,
        #     scale,
        # )
        # o = memory_efficient_attention_forward(
        #     q.view(-1, 1, self.tp_q_head_num, self.head_dim),
        #     k.repeat_interleave(group_size, -2),
        #     v.repeat_interleave(group_size, -2),
        #     op=FwOp
        # )
        o = q
        o = o.view(-1, self.tp_q_head_num * self.head_dim)
        GlobalTimer.stop("attn_comp")
        return o

    def deft_node_unpaged_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        o = torch.zeros_like(q)
        q = q.view(-1, self.tp_q_head_num, self.head_dim)
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        self.store_kv_cache(k, v, input_metadata)
        tree = get_global_tree_cache()
        kv = tree.get_kv_tree(self.layer_id)
        k, v = kv[:, 0], kv[:, 1]  # (seq_len, head_num, head_dim)
        GlobalTimer.stop("attn_mem")
        GlobalTimer.start("attn_comp")
        tree_metadata = get_global_tree_metadata()
        assert tree_metadata is not None
        PerfMetrics.update_KV_IO(
            kv.shape[0], self.tp_q_head_num * self.head_dim
        )

        deft_node_unpaged(
            q,
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            tree_metadata.node_kv_offset,
            tree_metadata.node_kv_len,
            tree_metadata.node_q,
            tree_metadata.node_q_offset,
            tree_metadata.node_q_len,
        )

        GlobalTimer.stop("attn_comp")
        return o

    def deft_flatten_unpaged_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        GlobalTimer.start("attn_mem")
        o = torch.zeros_like(q)
        q = q.view(-1, self.tp_q_head_num, self.head_dim)
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        self.store_kv_cache(k, v, input_metadata)
        tree = get_global_tree_cache()
        kv = tree.get_kv_tree(self.layer_id)
        k, v = kv[:, 0], kv[:, 1]  # (seq_len, head_num, head_dim)
        GlobalTimer.stop("attn_mem")
        GlobalTimer.start("attn_comp")
        tree_metadata = get_global_tree_metadata()
        assert tree_metadata is not None
        PerfMetrics.update_KV_IO(
            tree_metadata.total_kv_len,
            self.tp_q_head_num * self.head_dim,
        )
        PerfMetrics.update_Mask_IO(tree_metadata.total_kv_len)

        deft_flatten_unpaged(
            q,
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            tree_metadata.block_len,
            tree_metadata.block_q,
            tree_metadata.block_q_cnts,
            tree_metadata.block_q_offset,
            tree_metadata.block_bitmasks,
            tree_metadata.block_lens,
        )

        GlobalTimer.stop("attn_comp")
        return o

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)
        # gc.collect(2)
        # TODO(jinwei): placing both DeFT attention and Token attention under the class of DeFTAttention is kind of puzzling. Maybe we should rename it later.
        if input_metadata.forward_mode == ForwardMode.PREFILL:
            o = self.prefill_forward_triton(q, k, v, input_metadata)
        elif (
            input_metadata.forward_mode == ForwardMode.TREE_DECODE_FLATTEN
        ):  # DeFT Attention Kernel
            o = self.deft_flatten_forward(q, k, v, input_metadata)
        elif (
            input_metadata.forward_mode == ForwardMode.TREE_DECODE_NODE
            or input_metadata.forward_mode == ForwardMode.TREE_DECODE_INDEX_NODE
        ):  # DeFT Attention Kernel
            o = self.deft_node_forward(q, k, v, input_metadata)
        elif (
            input_metadata.forward_mode == ForwardMode.DECODE
        ):  # Token Attention Kernel in SGLang
            o = self.radix_attention_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.UNPAGED_FD:
            o = self.flash_decoding_unpaged_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.UNPAGED_MEDUSA:
            o = self.medusa_unpaged_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.UNPAGED_DEFT_NODE:
            o = self.deft_node_unpaged_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.UNPAGED_DEFT_FLATTEN:
            o = self.deft_flatten_unpaged_forward(q, k, v, input_metadata)
        else:
            raise NotImplementedError(
                f"Unsupported forward mode: {input_metadata.forward_mode}"
            )
        # gc.collect(2)
        return o

    def store_kv_cache(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        # key_buffer = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
        # value_buffer = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
        # if input_metadata.kv_updater.cache_loc is not None:
        #     cache_loc = input_metadata.kv_updater.cache_loc
        #     key_buffer[cache_loc] = cache_k
        #     value_buffer[cache_loc] = cache_v
        # else:
        input_metadata.kv_updater.update(self.layer_id, cache_k, cache_v)
