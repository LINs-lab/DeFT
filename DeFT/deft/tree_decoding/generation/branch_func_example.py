import torch
from typing import Optional

# from torch import nn
# from tqdm import tqdm
from deft.data_loader import ExecuteTree
from deft.model_runner import ModelRunner
# from deft.tree_decoding.tree_cache import TreeMetadata, TreeCache, TreeNode


@torch.no_grad()
def example_branch_Func1_SimpleTree(
    model: ModelRunner,
    iter: int,
    max_gen_len: int,
    width: int,
    depth: int,
    logits: torch.Tensor,
    **kwargs: dict,
) -> bool:
    stop_criteria = False
    token_tree_size = width
    assert model.tree.root is not None
    if iter + 1 == max_gen_len:
        stop_criteria = True
        for leaf in model.tree.leaves.values():
            model.tree.output_branch(dstnode=leaf)
        # # output root node
        # print(
        #     f"Root Tokens(length={len(model.tree.root.token_ids)})={model.tree.root.token_ids}\n"
        # )
        return stop_criteria

    if iter == 0:  # prefill
        # branch to token tree
        # topk_logits, topk_tokenIDs=logits[0].topk(token_tree_size, dim=-1)
        logprob = torch.log(logits[0])
        topk_logprob, topk_tokenIDs = logprob.topk(token_tree_size, dim=-1)
        # print(f"{logits[0][topk_tokenIDs]} {topk_logprob}")
        model.tree.branch(model.tree.root, token_tree_size)
        leaves = list(model.tree.leaves.values())
        leaf_cnt = 0
        for leaf in leaves:  # append tokens
            leaf.append_token(
                int(topk_tokenIDs[leaf_cnt].item()),
                logprob=topk_logprob[leaf_cnt].item(),
            )
            leaf_cnt += 1
    else:
        # append tokens
        # greedy
        leaves = list(model.tree.leaves.values())
        logprob = torch.log(logits)
        token_ids = torch.argmax(logprob, dim=1)
        for leaf in leaves:
            token_id = int(token_ids[model.tree.leaf_to_q[leaf.id]].item())
            leaf.append_token(
                token_id,
                logprob=logprob[model.tree.leaf_to_q[leaf.id], token_id].item(),
            )

    return stop_criteria


# def example_branch_Func2_BeamSearch(
#     model: ModelRunner,
#     iter,
#     max_gen_len,
#     width,
#     depth,
#     logits,
#     execution_graph=None,
# ) -> bool:
#     # We sample 2 * beam_width candidates to make sure that with high
#     # probability we can get `beam_width` candidates in addition to
#     # for details. See also HF reference:
#     # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
#     #

#     # 1) We don't set depth as our tree decoding constraints.
#     # 2) early stopping: we set early_stopping == True for sampling.  Later version with sampler could support "early_stopping == Never or early_stopping == Flase".
#     # TODO(jinwei): support selection of early stopping in simpler. Refer to vLLM: https://github.com/vllm-project/vllm/blob/cc74b2b232070f74d8765a5eefa49ae93ee45490/vllm/engine/output_processor/single_step.py#L234

#     def _beam_search_sample(
#         tree: TreeCache,
#         is_prompt: bool,
#         beam_width: int,
#         logprobs: torch.Tensor,
#     ):
#         # We sample 2 * beam_width candidates to make sure that with high probability we can get `beam_width` candidates in addition to the finished sequences for the next iteration.
#         results = []
#         if is_prompt:  # Prompt phase.
#             parent_ids = [0] * (2 * beam_width)
#             next_token_logprobs, next_token_ids = torch.topk(
#                 logprobs[0], 2 * beam_width, dim=-1, largest=True, sorted=True
#             )
#             next_token_ids = next_token_ids.tolist()
#             next_token_logprobs = next_token_logprobs.tolist()
#         else:  # Generation phase(Decoding Stage)
#             # leaf nodes
#             leaves = list(tree.leaves.values())
#             cumulative_logprobs = [lf.cumulative_logprob for lf in leaves]
#             cumulative_logprobs = torch.tensor(
#                 cumulative_logprobs, dtype=torch.float, device=logprobs.device
#             )
#             cumulative_logprobs = logprobs + cumulative_logprobs[
#                 :, None
#             ].expand_as(logprobs)
#             next_token_logprobs, topk_ids = torch.topk(
#                 cumulative_logprobs.flatten(), 2 * beam_width
#             )
#             topk_ids = topk_ids.tolist()
#             vocab_size = logprobs.size(-1)
#             leaves_ids = [i // vocab_size for i in topk_ids]
#             parent_ids = [leaves[lid].id for lid in leaves_ids]
#             next_token_ids = [i % vocab_size for i in topk_ids]
#             next_token_logprobs = next_token_logprobs.tolist()

#         # sort sample results

#         # return sample results
#         results.append((next_token_ids, next_token_logprobs, parent_ids))

#         return results

#     def beam_output_processor(
#         tree: TreeCache,
#         beam_width: int,
#         sample_res,  # List of (next_token_ids, next_token_logprobs, parent_ids)
#         eos_token_id: int,
#         seq_len: int,
#         early_stopping: Optional[Union[bool, str]] = True,
#     ):
#         # we set early_stopping == True for sampling.

#         def get_beam_search_score(
#             cumulative_logprob: float,
#             seq_len: int,
#             length_penalty: float = 1.0,
#         ) -> float:
#             """Calculate the beam search score with length penalty.
#             Adapted from
#             https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
#             """
#             return cumulative_logprob / (seq_len**length_penalty)

#         # get beam search score and sorted nodes
#         batch_size = len(
#             sample_res
#         )  # how many trees can be batched. only support one tree in this version.
#         assert (
#             batch_size == 1
#         )  # Multi-tree batching could be supported in the future.
#         next_beam_scores = torch.zeros((batch_size, beam_width), dtype=float)
#         next_beam_tokens = torch.zeros((batch_size, beam_width), dtype=int)
#         next_beam_pids = torch.zeros((batch_size, beam_width), dtype=int)
#         parent_beam_mapping = {}
#         for tree_res in sample_res:  # for each tree, the sample result is (next_token_ids, next_token_logprobs, parent_ids)
#             # sort based on next_token_logprobs
#             sorted_indices = sorted(
#                 range(len(tree_res[1])),
#                 key=lambda i: tree_res[1][i],
#                 reverse=True,
#             )
#             next_token_logprobs = sorted(tree_res[1], reverse=True)
#             next_token_ids = [tree_res[0][i] for i in sorted_indices]
#             parent_ids = [tree_res[2][i] for i in sorted_indices]

#             # # debug
#             # print(f"next_token_ids: {next_token_ids}")
#             # print(f"next_token_logprobs: {next_token_logprobs}")
#             # print(f"parent_ids: {parent_ids}")

#             beam_idx = 0
#             for next_token_id, next_token_logprob, parent_id in zip(
#                 next_token_ids, next_token_logprobs, parent_ids
#             ):
#                 if next_token_id == eos_id:
#                     tree.output_branch(dstnode=tree.nodes[parent_id])
#                 else:
#                     # record token, score, parent_seq_id until we get beam_width non-eos ids.
#                     # add next predicted token since it is not eos_token
#                     next_beam_scores[0, beam_idx] = next_token_logprob
#                     next_beam_tokens[0, beam_idx] = next_token_id
#                     next_beam_pids[0, beam_idx] = parent_id
#                     if parent_id not in parent_beam_mapping:
#                         parent_beam_mapping[parent_id] = []
#                     parent_beam_mapping[parent_id].append(beam_idx)
#                     beam_idx += 1
#                 if beam_idx == beam_width:
#                     break
#         # stop criteria
#         if len(tree.all_finished_seqs) >= beam_width:  # early stop
#             # sort and select branches with beam_width highest scores
#             tree.all_finished_seqs.sort(
#                 key=lambda x: get_beam_search_score(
#                     x.cumulative_logprob, x.get_len()
#                 ),
#                 reverse=True,
#             )
#             tree.all_finished_seqs = tree.all_finished_seqs[:beam_width]
#             # Note(jinwei): we comment here to prevent the early stopping, to make sure the max gen length could be reached.
#             # return True

#         # get the list of nodes to prune and branch.
#         last_iter_leaves = tree.leaves.keys()
#         next_iter_leaves = parent_beam_mapping.keys()
#         prune_nodes = list(set(last_iter_leaves) - set(next_iter_leaves))
#         # # debug
#         # print(f" parent_beam_mapping={parent_beam_mapping}\n")
#         # print(f"last_iter_leaves={last_iter_leaves}\n",
#         # f"next_iter_leaves={next_iter_leaves}\n",
#         # f"prune_nodes={prune_nodes}\n")

#         # branch/prune operation
#         leaves = list(model.tree.leaves.values())
#         for leaf in leaves:
#             l_id = leaf.id
#             if l_id in next_iter_leaves:
#                 child_num = len(parent_beam_mapping[l_id])
#                 if child_num == 1:  # append
#                     b_idx = parent_beam_mapping[l_id][0]
#                     leaf.append_token(next_beam_tokens[0, b_idx].item())
#                     leaf.cumulative_logprob = next_beam_scores[0, b_idx]
#                 elif child_num > 1:  # branch
#                     children_nodes = tree.branch(leaf, child_num)
#                     # append tokens
#                     assert len(children_nodes) == child_num
#                     child_cnt = 0
#                     for b_idx in parent_beam_mapping[l_id]:
#                         children_nodes[child_cnt].append_token(
#                             next_beam_tokens[0, b_idx].item()
#                         )
#                         children_nodes[
#                             child_cnt
#                         ].cumulative_logprob = next_beam_scores[0, b_idx]
#                         child_cnt += 1
#                 else:
#                     raise RuntimeError
#             elif l_id in prune_nodes:  # prune
#                 # assert model.tree.nodes[l_id] in model.tree.leaves

#                 deleted_nodes = model.tree.cut(model.tree.nodes[l_id])
#                 # for node in deleted_nodes:
#                 #     model.tree.nodes.pop(node.id)
#             else:
#                 raise RuntimeError
#         return False

#     # For beam search, beam width=width.
#     beam_width = width
#     vocab_size = logits.shape[-1]
#     eos_id = model.model_config.hf_config.eos_token_id
#     is_prompt = True if iter == 0 else False
#     next_token_scores = torch.log(logits)  # to get logprob

#     # beam sample
#     sample_res = _beam_search_sample(
#         tree=model.tree,
#         is_prompt=is_prompt,
#         beam_width=beam_width,
#         logprobs=next_token_scores,
#     )

#     # debug: check sample results
#     if iter == 0:
#         print(
#             f"For prefill, Sample results={sample_res[0]}\n",
#             f"Token candidates={sample_res[0][0]}\n",
#             f"Cumulative log prob={sample_res[0][1]}\n",
#             f"Parents ids={sample_res[0][2]}\n",
#         )
#     # beam processor for selection: include beams to branch and prune.
#     seq_len = (
#         len(model.tree.root.token_ids) + iter
#     )  # get current length of each branch in the decoding tree
#     stop_criteria = beam_output_processor(
#         tree=model.tree,
#         beam_width=beam_width,
#         sample_res=sample_res,
#         eos_token_id=eos_id,
#         seq_len=seq_len,
#     )
#     assert len(model.tree.leaves) == beam_width, (
#         f"The number of leaves should be beam width= {beam_width}, but it is {len(model.tree.leaves)}"
#     )
#     if iter == max_gen_len - 1:
#         for leaf in model.tree.leaves.values():
#             model.tree.output_branch(dstnode=leaf)
#     return stop_criteria


def example_branch_Func3_FromTreeTemplate(
    model: ModelRunner,
    iter: int,
    max_gen_len: int,
    width: int,
    depth: int,
    logits: torch.Tensor,
    execution_graph: Optional[ExecuteTree] = None,
) -> bool:
    # execution_graph: model.tree.template_tree
    # which contains 'branch_record' and 'prune_record'
    assert execution_graph is not None
    branch_pairs = execution_graph.branch_record.get(
        iter, {}
    )  # if iter is not a key, return an empty dic
    prune_nodes = execution_graph.prune_record.get(
        iter, []
    )  # if iter is not a key, return an empty list
    stop_criteria = False
    # Todo(jinwei): deal with a corner case for output. When the root node is going to prompt, we output all branches before prune it. We detect whether root node is in prune_nodes and output tokens in the branch before pruning the root.
    root_ID = 0
    if root_ID in prune_nodes:
        stop_criteria = True
        for leaf in model.tree.leaves.values():
            model.tree.output_branch(dstnode=leaf)

    assert model.tree.root is not None
    # branch
    parent_ids = branch_pairs.keys()  # get parent id
    if iter == 0:
        leaves = [model.tree.root]
    else:
        leaves = list(model.tree.leaves.values())
    for leaf in leaves:
        l_id = leaf.id
        if l_id in parent_ids:
            p_node = model.tree.nodes[l_id]
            # assert p_node in model.tree.leaves
            children_ids = branch_pairs[l_id]
            branch_width = len(children_ids)
            assert branch_width > 0
            if (
                iter == 0
            ):  # when iter=0, the tree only has a root with no leaves
                q_idx = 0
            else:
                q_idx = model.tree.leaf_to_q[
                    p_node.id
                ]  # get q index for the parent node
            topk_logits, topk_tokenIDs = logits[q_idx].topk(
                branch_width, dim=-1
            )  # generate top k tokens, where k=width
            children_nodes = model.tree.branch(p_node, branch_width)
            # append tokens
            assert len(children_nodes) == branch_width
            for child_cnt in range(branch_width):
                children_nodes[child_cnt].append_token(
                    int(topk_tokenIDs[child_cnt].item())
                )
        # prune
        elif l_id in prune_nodes:
            # assert model.tree.nodes[l_id] in model.tree.leaves

            _ = model.tree.cut(model.tree.nodes[l_id], record_deleted=True)
            # for node in deleted_nodes:
            #     model.tree.nodes.pop(node.id)
        else:  # greedy generation for leaves
            token_ids = torch.argmax(logits, dim=1)
            leaf.append_token(
                int(token_ids[model.tree.leaf_to_q[leaf.id]].item())
            )
    # output the branches in the last itertaion or when the entire tree is pruned.
    if iter == max_gen_len - 1:
        for leaf in model.tree.leaves.values():
            model.tree.output_branch(dstnode=leaf)
        # stop the decode
        stop_criteria = True

    return stop_criteria


def example_branch_Func4_SpeculativeDecoding(
    model: ModelRunner,
    iter: int,
    max_gen_len: int,
    width: int,
    depth: int,
    logits: torch.Tensor,
    execution_graph: Optional[ExecuteTree] = None,
) -> bool:
    assert execution_graph is not None
    assert execution_graph.accepted_len_list is not None
    stop_criteria = False
    last_step = len(execution_graph.accepted_len_list)
    token_tree_size = execution_graph.node_num
    if iter == last_step:
        stop_criteria = True
        # sequeeze_node = model.tree.nodes[1].get_len()
        # print(f"sequeezed node len={sequeeze_node}\n")
        for leaf in model.tree.leaves.values():
            model.tree.output_branch(dstnode=leaf)
        # # output root node
        # print(
        #     f"Root Tokens(length={len(model.tree.root.token_ids)})={model.tree.root.token_ids}\n"
        # )
        return stop_criteria
    verified_num = execution_graph.accepted_len_list[iter]
    if iter == 0:  # prefill
        # branch to token tree
        topk_logits, topk_tokenIDs = logits[0].topk(token_tree_size, dim=-1)
        model.tree.branch(model.tree.root, token_tree_size)
        leaves = list(model.tree.leaves.values())
        leaf_cnt = 0
        for leaf in leaves:  # append tokens
            leaf.append_token(topk_tokenIDs[leaf_cnt].item())
            leaf_cnt += 1
        # Todo: Use KV of top verified_num of root to simulate KV of verified tokens, then append them to past seq KV(root).

    else:
        # method for mocking: always keep all leaves and squeeze the verified tokens to the root.
        leaves = list(model.tree.leaves.values())
        assert len(leaves) == token_tree_size
        if model.tree.use_paged_memory:
            kv_len_before = len(model.tree.root.kv_indices)
        else:
            kv_len_before = model.tree.root.kv_data.get_key_buffer(0).shape[0]
        # topk_logits, topk_tokenIDs=logits[0].topk(token_tree_size, dim=-1)
        for i in range(0, len(leaves)):
            if i < verified_num:
                model.tree.merge_nodes(
                    model.tree.root, leaves[i], pruneB_flag=False
                )
            # release their kv  indices
            # if model.tree.use_paged_memory:
            #     model.tree.token_to_kv_pool.free(torch.tensor(leaves[i].kv_indices))

        # delete other node's KV index

        if model.tree.use_paged_memory:
            kv_len_after = len(model.tree.root.kv_indices)
        else:
            kv_len_after = model.tree.root.kv_data.get_key_buffer(0).shape[0]
        diff = kv_len_after - kv_len_before
        for leaf in leaves:
            model.tree.reset_node_KV(leaf, diff)
        assert (
            kv_len_before + verified_num == kv_len_after
        ), f"length of KV after squeeze of token tree is incorrect!KV len before={kv_len_before},verified_num={verified_num}, while kv_len_after={kv_len_after}\n"

    return stop_criteria
