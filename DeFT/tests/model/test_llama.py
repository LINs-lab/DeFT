# type: ignore
import time

import numpy as np
import torch
from deft.model_runner import ModelRunner
from deft.model_runner import ForwardMode
from deft.tree_decoding.tree_cache import TreeMetadata
from deft.model_config import ModelConfig

# from deft.hf_transformers_utils import get_tokenizer
import argparse

# from Tree_Decoding.utils import load, download_url, load_jsonl


def inference_interface(
    model: ModelRunner, mode: ForwardMode, prompt_len, node_len, width, depth
):
    # batch_size = 1
    # Prepare data
    input_ids = np.arange(5, prompt_len + 5)
    input_ids = torch.tensor(input_ids).cuda()

    def init_data(model: ModelRunner, batch_size, input_len):
        req_pool_indices = model.req_to_token_pool.alloc(batch_size)
        assert req_pool_indices is not None
        seq_lens = torch.full(
            (batch_size,), input_len, dtype=torch.int32, device="cuda"
        )
        prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        position_ids_offsets = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        # NOTE(jinwei): for tree decoding, the KV pool allocation size could be reset.
        out_cache_loc = model.token_to_kv_pool.alloc(batch_size * input_len)
        assert out_cache_loc is not None
        for i in range(batch_size):
            req_idx = req_pool_indices[i].item()
            model.req_to_token_pool.req_to_token[req_idx, :input_len] = (
                out_cache_loc[i * input_len : (i + 1) * input_len]
            )

        return (
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            out_cache_loc,
        )

    def init_tree_data(model: ModelRunner, prompt_len):
        req_pool_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        position_ids_offsets = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        out_cache_loc = model.token_to_kv_pool.alloc(prompt_len)
        assert out_cache_loc is not None
        model.tree.init_prompt(input_ids, out_cache_loc)
        return (
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            out_cache_loc,
        )

    def prefill():
        logits, _ = model.forward_prefill(
            input_ids,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            out_cache_loc,
            False,
        )
        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        # print("prefill logits", logits, logits.shape)

    # NOTE(jinwei): add branch controller on top of decode.
    def decode(forward_mode: ForwardMode):
        # (
        #     out_cache_loc,
        #     out_cache_cont_start,
        #     out_cache_cont_end,
        # ) = model.token_to_kv_pool.alloc_contiguous(batch_size)

        leaf_to_q: dict[int, int] = {}
        leaf_cnt = 0

        for leaf in sorted(model.tree.leaves.values(), key=lambda x: x.id):
            leaf_to_q[leaf.id] = leaf_cnt
            leaf_cnt += 1

        token_ids = [0 for _ in range(leaf_cnt)]
        positions = [0 for _ in range(leaf_cnt)]
        # NOTE(jinwei): we need to check the allocation in tests.
        out_cache_loc = model.token_to_kv_pool.alloc(leaf_cnt)
        assert out_cache_loc is not None
        for leaf in model.tree.leaves.values():
            q_idx = leaf_to_q[leaf.id]
            token_ids[q_idx] = leaf.token_ids[-1]
            positions[q_idx] = leaf.positions[-1]
            leaf.append_index(out_cache_loc[q_idx].item())

        tree_metadata = TreeMetadata.from_tree_cache(model.tree)

        # model.req_to_token_pool.req_to_token[req_pool_indices, seq_lens] = out_cache_loc
        # seq_lens.add_(1)
        result, t = model.forward_tree_decode(
            forward_mode,
            torch.tensor(token_ids).cuda().reshape(-1),
            torch.tensor(positions).cuda().reshape(-1),
            out_cache_loc,
            True,
            tree_metadata,
        )
        logits = result[0]

        # print('')
        # logits_ref = model.forward_tree_decode(
        #     ForwardMode.DECODE,
        #     torch.tensor(token_ids).cuda().reshape(-1),
        #     torch.tensor(positions).cuda().reshape(-1),
        #     out_cache_loc,
        #     True,
        #     tree_metadata,
        # )[0]

        prob_out = torch.softmax(logits, dim=-1)
        token_ids = torch.argmax(prob_out, dim=1)

        # logits = torch.softmax(logits.to(torch.float64), dim=-1)
        # logits_ref = torch.softmax(logits_ref.to(torch.float64), dim=-1)
        # print('')
        # print(torch.nn.functional.mse_loss(logits, logits_ref))
        # print('')

        return token_ids, leaf_to_q, t

        # token_ids = token_ids.detach().cpu().numpy()
        # print("decode", i, logits)

    # Warm up
    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    ) = init_tree_data(model, prompt_len)

    nodes = [model.tree.root]

    # for i in range(batch_size):
    #     req_idx = req_pool_indices[i].item()
    #     model.token_to_kv_pool.free(
    #         model.req_to_token_pool.req_to_token[req_idx, : seq_lens[i]]
    #     )
    # model.req_to_token_pool.free(req_pool_indices)

    # Benchmark
    start_time = prefill_start_time = time.time()

    # (
    #     req_pool_indices,
    #     seq_lens,
    #     prefix_lens,
    #     position_ids_offsets,
    #     out_cache_loc,
    # ) = init_batch_data(model, batch_size, input_len)

    prefill()

    reserve = 1

    print(f"prefill cost: {(time.time() - prefill_start_time) * 1000:.2f} ms")

    step_time = []
    # NOTE(jinwei): simple tree case. Move to branch controller later.
    for i in range(depth):
        step_start = time.time()
        torch.cuda.synchronize()
        # leaves = model.tree.branch(node, width)
        for node in nodes:
            model.tree.branch(node, width)
        leaves = list(model.tree.leaves.values())
        for leaf in leaves:
            leaf.append_token(5)
        for _ in range(node_len - 1):
            token_ids, leaf_to_q, t = decode(mode)
            step_time.append(t)
            for leaf in leaves:
                leaf.append_token(token_ids[leaf_to_q[leaf.id]].item())

        token_ids, leaf_to_q, t = decode(mode)
        step_time.append(t)
        for leaf in leaves[reserve:]:
            deleted_nodes = model.tree.cut(leaf)
            # for node in deleted_nodes:
            #     model.token_to_kv_pool.free(node.kv_indices)
            #     model.tree.nodes.pop(node.id)
        nodes = leaves[:reserve]

        step_end = time.time()
        torch.cuda.synchronize()
        print(f"step {i} cost: {(step_end - step_start) * 1000:.2f} ms")

    deleted_nodes = []
    for node in nodes:
        deleted_nodes.extend(model.tree.cut(node))
    # for node in deleted_nodes:
    #     model.token_to_kv_pool.free(node.kv_indices)
    #     model.tree.nodes.pop(node.id)
    end_time = time.time()

    print(f"total cost: {(end_time - start_time) * 1000:.2f} ms")
    print(f"decode time: {(model.decode_time * 1000):.2f} ms")
    print(f"step time: {step_time}")


def main(args: argparse.Namespace):
    model_path = args.model
    tp_rank = 0
    tp_size = 1
    nccl_port = 28888
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(model_config, 0.8, tp_rank, tp_size, nccl_port)
    # tokenizer = get_tokenizer(model_path)
    mode = None
    if args.mode == "tree":
        mode = ForwardMode.TREE_DECODE
    elif args.mode == "seq":
        mode = ForwardMode.DECODE
    assert mode is not None
    inference_interface(model, mode, 1000, 1000, 10, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--mode", choices=["tree", "seq"], default="seq")
    # parser.add_argument("--data_root", type=str, default="data/")
    # # load tree templates
    # parser.add_argument("--dataset", type=str, default=None)
    # # parser.add_argument("--enable_TreeDecoding", action="store_true")
    # parser.add_argument("--Decoding_Pattern", choices=["Naive_seq_based", "Flash_decoding","Flash_decoding_TreeKV", "CausalMasked_tree", "DeFT"], default="Naive_seq_based")
    # # define the way that we do the branch controlling below.
    # parser.add_argument("--Branch_contoller", choices=["Simple_Tree", "Beam_Search","Random_Tree", "Practical_Tree"], default="Simple_Tree")
    # parser.add_argument("--Sampling_method", choices=["Logits_topK", "Accu_probs_topK","User_defined" ], default="Logits_topK")
    # parser.add_argument("--max_depth", type=int, default=10)
    # parser.add_argument("--max_width", type=int, default=10)
    # parser.add_argument("--max_seq_len", type=int, default=500)
    # parser.add_argument("--tree_idx", type=int, default=0)
    # parser.add_argument("--save_to_file", type=str, default=None)
    args = parser.parse_args()

    main(args)
