"""Force a specific tree decoding process based on user-defined branch function"""

from typing import Optional
import torch
# import gc

# import numpy as np
import time
from tqdm import tqdm
from deft.tree_decoding.branch_controller import Branch_Controller
from deft.model_runner import ModelRunner, ForwardMode
from deft.tree_decoding.tree_cache import TreeMetadata, KVCacheUpdater
from deft.tree_decoding.timer import GlobalTimer
from deft.tree_decoding.perf_metrics import PerfMetrics
from deft.data_loader import ExecuteTree
from transformers import AutoTokenizer
from typing import Tuple


def tree_generate(
    model: ModelRunner,
    mode: ForwardMode,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    max_seq_len: int,
    width: int,
    depth: int,
    branch_controller: Branch_Controller,
    tree_template: Optional[ExecuteTree],
    output_file: Optional[str] = None,
    perf_metrics: Optional[PerfMetrics] = None,
) -> None:
    # batch_size = 1
    # Prepare data
    # gc.disable()

    prompt_len = prompt_ids.shape[1]
    # TODO(jinwei): we should process the case when there are more than one prompts in the later version.
    input_ids = prompt_ids[0]  # extract the first prompt as an example
    max_gen_len = max_seq_len - prompt_len
    # node_len=max_gen_len/depth

    def init_tree_data(
        model: ModelRunner, prompt_len: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, KVCacheUpdater
    ]:
        seq_lens = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        position_ids_offsets = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        kv_updater = model.tree.init_prompt(input_ids)
        root = model.tree.root
        assert root is not None
        if model.use_paged_memory:
            req = model.tree.leaf_to_req[root.id]
            req_pool_indices = torch.tensor(
                [req], dtype=torch.int32, device="cuda"
            )
        else:
            req_pool_indices = torch.tensor(
                [0], dtype=torch.int32, device="cuda"
            )
        return (
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            kv_updater,
        )

    def prefill() -> torch.Tensor:
        logits, _ = model.forward_prefill(
            input_ids,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            kv_updater,
            False,
        )
        prob_out = torch.softmax(logits, dim=-1)
        return prob_out

        # predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        # predict_ids = predict_ids.detach().cpu().numpy()

        # print("prefill logits", logits, logits.shape)

    # NOTE(jinwei): add branch controller on top of decode.
    def decode(
        forward_mode: ForwardMode,
    ) -> Tuple[torch.Tensor, dict[int, int], float]:
        GlobalTimer.start("prepare")
        GlobalTimer.start("positions")
        leaf_to_q: dict[int, int] = {}
        leaf_cnt = 0

        for leaf in sorted(model.tree.leaves.values(), key=lambda x: x.id):
            leaf_to_q[leaf.id] = leaf_cnt
            leaf_cnt += 1

        token_ids = [0 for _ in range(leaf_cnt)]
        positions = [0 for _ in range(leaf_cnt)]
        # NOTE(jinwei): we need to check the allocation in tests.
        # out_cache_loc = model.token_to_kv_pool.alloc(leaf_cnt)
        # assert out_cache_loc is not None
        kv_updater = model.tree.alloc()
        for leaf in model.tree.leaves.values():
            q_idx = leaf_to_q[leaf.id]
            token_ids[q_idx] = leaf.token_ids[-1]
            positions[q_idx] = leaf.positions[-1]

        use_deft = forward_mode in [
            ForwardMode.TREE_DECODE_FLATTEN,
            ForwardMode.TREE_DECODE_NODE,
            ForwardMode.TREE_DECODE_INDEX_NODE,
            ForwardMode.UNPAGED_DEFT_FLATTEN,
            ForwardMode.UNPAGED_DEFT_NODE,
        ]
        GlobalTimer.stop("positions")
        GlobalTimer.start("tree_metadata")
        if use_deft:
            if model.use_tree_index:
                tree_metadata = TreeMetadata.from_tree_cache_node(model.tree)
            else:
                tree_metadata = TreeMetadata.from_tree_cache(model.tree)
        else:
            tree_metadata = None
        GlobalTimer.stop("tree_metadata")
        # model.req_to_token_pool.req_to_token[req_pool_indices, seq_lens] = out_cache_loc
        # seq_lens.add_(1)
        # GlobalTimer.stop("prepare")
        # GlobalTimer.start("forward")
        # result, t, kv_len, q_len, seq_total = model.forward_tree_decode(
        result, t = model.forward_tree_decode(
            forward_mode,
            torch.tensor(token_ids).cuda().reshape(-1),
            torch.tensor(positions).cuda().reshape(-1),
            kv_updater,
            True,
            tree_metadata,
        )
        # GlobalTimer.stop("forward")
        GlobalTimer.start("branch")
        logits = result[0]  # last_logits

        # print('')
        # logits_ref = model.forward_tree_decode(
        #     ForwardMode.DECODE,
        #     torch.tensor(token_ids).cuda().reshape(-1),
        #     torch.tensor(positions).cuda().reshape(-1),
        #     out_cache_loc,
        #     True,
        #     tree_metadata,
        # )[0]

        prob_out = torch.softmax(logits.float(), dim=-1) + 1e-6

        # token_ids = torch.argmax(prob_out, dim=1)

        # logits = torch.softmax(logits.to(torch.float64), dim=-1)
        # logits_ref = torch.softmax(logits_ref.to(torch.float64), dim=-1)
        # print('')
        # print(torch.nn.functional.mse_loss(logits, logits_ref))
        # print('')

        return prob_out, leaf_to_q, t  # , kv_len, q_len, seq_total

        # token_ids = token_ids.detach().cpu().numpy()
        # print("decode", i, logits)

    # Warm up
    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        kv_updater,
    ) = init_tree_data(model, prompt_len)

    # load tree templates as the execution graph of branch
    branch_controller.set_execution_graph(tree_templates=tree_template)

    # nodes = [model.tree.root]
    # Benchmark
    start_time = prefill_start_time = time.time()
    prob = prefill()
    # apply branch function for the prefill.
    stop = branch_controller.apply_branching(
        model=model,
        iter=0,
        max_gen_len=max_gen_len,
        width=width,
        depth=depth,
        logits=prob,
        execution_graph=branch_controller.tree_templates,
    )
    ttft = (time.time() - prefill_start_time) * 1000

    print(f"prob shape:{prob.shape}\n")

    # NOTE(jinwei): step function here for the decoding stage.
    if stop is False:
        for iter in tqdm(range(1, max_gen_len), desc="Processing iterations"):
            GlobalTimer.reset("decoding")
            GlobalTimer.reset("prepare")
            GlobalTimer.reset("forward")
            GlobalTimer.reset("branch")
            GlobalTimer.reset("attn_mem")
            GlobalTimer.reset("attn_comp")
            GlobalTimer.reset("traversal")
            GlobalTimer.reset("alloc")
            GlobalTimer.reset("positions")
            GlobalTimer.reset("tree_metadata")
            GlobalTimer.reset("input_metadata")

            GlobalTimer.start("decoding")

            torch.cuda.synchronize()
            step_start = time.time()

            # prob, leaf_to_q, t, kv_len, q_len, seq_total = decode(mode)
            prob, leaf_to_q, t = decode(mode)
            # step_time.append(t)
            # kv_lens.append(kv_len)
            # q_lens.append(q_len)
            # seq_totals.append(seq_total)
            model.tree.leaf_to_q = leaf_to_q
            # apply user-defined function to branch and prune.
            stop = branch_controller.apply_branching(
                model=model,
                iter=iter,
                max_gen_len=max_gen_len,
                width=width,
                depth=depth,
                logits=prob,
                execution_graph=branch_controller.tree_templates,
            )
            GlobalTimer.stop("branch")

            torch.cuda.synchronize()
            step_end = time.time()

            iter_cost = (step_end - step_start) * 1000

            GlobalTimer.stop("decoding")
            if perf_metrics is not None:
                perf_metrics.update(
                    iter_time=iter_cost,
                    prepare=GlobalTimer.get("prepare"),
                    forward=t * 1000,
                    branch=GlobalTimer.get("branch"),
                    attn_mem=GlobalTimer.get("attn_mem"),
                    attn_comp=GlobalTimer.get("attn_comp"),
                    traversal=GlobalTimer.get("traversal"),
                    alloc=GlobalTimer.get("alloc"),
                    positions=GlobalTimer.get("positions"),
                    tree_metadata=GlobalTimer.get("tree_metadata"),
                    input_metadata=GlobalTimer.get("input_metadata"),
                )
            if stop:
                break

    end_time = time.time()
    profiling_latency = (end_time - start_time) * 1000
    PerfMetrics.update_e2e_latency(profiling_latency)

    # # get path output
    # model.tree.print_finished_branches(tokenizer=tokenizer)

    # print(f"wall clock(including profiling) time: {profiling_latency:.2f} ms")
    # print(f"decode latency: {(model.decode_time * 1000):.2f} ms ")

    tree_token_number = model.tree.get_tree_token_number()
    generated_tokens = tree_token_number - prompt_len
    # Time Per Output Token
    perf_metrics.print_latency(
        prompt_len=prompt_len, generated_len=generated_tokens, ttft=ttft
    )

    # free/reset the tree
    model.tree.free()
