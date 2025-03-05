# type: ignore
"""Provide the example for the usage of branch controller to control the tree decoding process."""

# import time
# import numpy as np

# import torch
import sys

# import os
from typing import Optional
from deft.model_runner import ModelRunner, ForwardMode

from deft.tree_decoding.generation import (
    tree_generate,
    branch_func_example,
)
from deft.tree_decoding.branch_controller import Branch_Controller
from deft.model_config import ModelConfig
from deft.hf_transformers_utils import get_tokenizer
from deft import data_loader
from deft.tree_decoding.perf_metrics import PerfMetrics
import argparse
from deft.utils import load_jsonl
from deft.tree_decoding.tree_cache import (
    BLOCK_CONFIG,
    TRAVERSAL_CONFIG,
)
import random

sys.setrecursionlimit(10000)
random.seed(0)


def inference_interface(
    model: ModelRunner,
    mode: ForwardMode,
    tokenizer,
    input_ids,
    max_seq_len,
    width,
    depth,
    branch_controller,
    tree_template: Optional[data_loader.ExecuteTree] = None,
    output_file: Optional[str] = None,
    perf_metrics: Optional[PerfMetrics] = None,
):
    # NOTE(jinwei): we leave flexibility between inference_interface() and tree_generate().

    # tree generate: apply user-defined branch controller for tree decoding process.
    tree_generate.tree_generate(
        model=model,
        mode=mode,
        tokenizer=tokenizer,
        prompt_ids=input_ids,
        max_seq_len=max_seq_len,
        width=width,
        depth=depth,
        branch_controller=branch_controller,
        tree_template=tree_template,
        output_file=output_file,
        perf_metrics=perf_metrics,
    )

    # We can output the result here.


def prompt_prepare(
    tokenizer,
    prompt: str,
    pad_prompt: Optional[str] = None,
    device: str = "cuda",
    desired_seq_len: Optional[int] = None,
):
    # prompt = "USER: " + prompt + "\n\nASSISTANT: "
    print(f"Generate token ids for the prompt: {prompt}")

    if desired_seq_len is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        # Note(jinwei): when the desired seq len is much larger than original len of prompt, the genertion result might look strage.
        input_ids = tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=desired_seq_len,
            return_tensors="pt",
        ).input_ids
        seq_len = input_ids.shape[1]
        while seq_len < desired_seq_len:
            prompt = prompt + pad_prompt
            input_ids = tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=desired_seq_len,
                return_tensors="pt",
            ).input_ids
            seq_len = input_ids.shape[1]
        assert (
            seq_len >= desired_seq_len
        ), f"Required seq len={desired_seq_len}, but we get {seq_len}"

    print(f"Token ids for the prompt: {input_ids}")
    input_ids = input_ids.to(device)

    return input_ids


def main(args: argparse.Namespace):
    model_path = args.model
    model_config = ModelConfig(path=model_path)
    use_tree_index = args.mode == "tree_index"
    model = ModelRunner(
        model_config,
        0.8,
        use_paged_memory=args.mem == "paged",
        use_tree_index=use_tree_index,
    )
    tokenizer = get_tokenizer(model_path)
    # tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.padding_side = "left"
    mode = None
    if args.mem == "unpaged":
        if args.mode == "tree":
            mode = ForwardMode.UNPAGED_MEDUSA
        elif args.mode == "seq":
            mode = ForwardMode.UNPAGED_FD
        elif args.mode == "flatten":
            mode = ForwardMode.UNPAGED_DEFT_FLATTEN
        elif args.mode == "node":
            mode = ForwardMode.UNPAGED_DEFT_NODE
        elif args.mode == "node_chunk":
            raise NotImplementedError
    else:
        if args.mode == "tree":
            # mode = ForwardMode.TREE_DECODE
            raise NotImplementedError
        elif args.mode == "seq":
            mode = ForwardMode.DECODE
        elif args.mode == "flatten":
            mode = ForwardMode.TREE_DECODE_FLATTEN
        elif args.mode == "node":
            mode = ForwardMode.TREE_DECODE_NODE
        elif args.mode == "node_chunk":
            mode = ForwardMode.TREE_DECODE_NODE
            BLOCK_CONFIG["MAX_BLOCK_LEN"] = BLOCK_CONFIG["BLOCK_LEN"]
        elif args.mode == "tree_index":
            mode = ForwardMode.TREE_DECODE_INDEX_NODE
            BLOCK_CONFIG["MAX_BLOCK_LEN"] = BLOCK_CONFIG["BLOCK_LEN"]

    assert mode is not None

    if args.traversal == "dfs":
        TRAVERSAL_CONFIG["method"] = "dfs"
    elif args.traversal == "bfs_token":
        TRAVERSAL_CONFIG["method"] = "bfs_token"
    elif args.traversal == "bfs_node":
        TRAVERSAL_CONFIG["method"] = "bfs_node"
    else:
        raise NotImplementedError

    # initial branch controller. You can set any user-defined function.
    if args.Branch_controller == "Simple_Tree":
        branch_controller = Branch_Controller(
            branching_function=branch_func_example.example_branch_Func1_SimpleTree
        )  # a simple example.
    elif args.Branch_controller == "Practical_Tree":
        branch_controller = Branch_Controller(
            branching_function=branch_func_example.example_branch_Func3_FromTreeTemplate
        )  # a practical example.
    elif args.Branch_controller == "Beam_Search":
        branch_controller = Branch_Controller(
            branching_function=branch_func_example.example_branch_Func2_BeamSearch
        )  # a beam search example.
    elif args.Branch_controller == "Speculative_Decoding":
        branch_controller = Branch_Controller(
            branching_function=branch_func_example.example_branch_Func4_SpeculativeDecoding
        )
    else:
        raise NotImplementedError

    # Prompt example.
    # Q83 = "Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words."
    Q_reasoner = """
You are a teacher. Below are some questions, reference answers and the answers from students. Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Reference answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = 18 every day at the farmer's market. Student answer: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. This means she uses 3 + 4 = 7 eggs every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. So she sells (16 - 7) * $2 = $6 worth of eggs every day. The answer is 6. Question: Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat in 4 weeks? Reference answer: She eats 3 eggs every day and there are 7 days in a week so she eats 3*7 = 21 eggs a week After 4 weeks she will have eaten 4*21 = 84 eggs There are 12 eggs in 1 dozen and she'll eat 84 eggs so that's 84/12 = 7 dozen eggs. Student answer: Claire makes a 3 egg omelet every morning. In one week she will eat 3 * 7 = 21 eggs. In 4 weeks she will eat 4 * 21 = 84 eggs. The answer is 84. Question: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs twice as much, how many dollars are the boots? Reference answer: The second pair of heels costs 33 * 2 = $66. The heels together cost 66 + 33 = $99. The boots cost $5 more than both pairs of heels together, so the boots cost 99 + 5 = $104. Student answer: We know that one pair of heels costs $33 and the other pair costs twice as much. This means that the other pair costs $33 * 2 = $66. Together, the two pairs of heels cost $33 + $66 = $99. The boots cost five dollars less than the heels, so the boots cost $99 - $5 = $94. The answer is $94. Question: Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay? Reference answer: The discount on the radiator was 400*.8=$320 So he paid 400-320=$80 The mechanic charges 3*50=$150 So in total he paid 80+150=$230 Student answer: The cost for a new radiator is $400. He got it at a junk shop for 80% off, so he paid $400 * 0.8 = $320 for the radiator. The mechanic charged him $50 per hour for 3 hours, so he paid $50 * 3 = $150 for the labor. In total, he paid $320 + $150 = $470. The answer is 470. Please summarize the mistakes in a short sentence for the question. At the end, please make a brief list of criteria. Make sure they are general and not specific to these questions so that others can grade the answers for other answers by following these criteria.
"""
    pad_prompt = """
Reference answer: The discount on the radiator was 400*.8=$320 So he paid 400-320=$80 The mechanic charges 3*50=$150 So in total he paid 80+150=$230 Student answer: The cost for a new radiator is $400. He got it at a junk shop for 80% off, so he paid $400 * 0.8 = $320 for the radiator. The mechanic charged him $50 per hour for 3 hours, so he paid $50 * 3 = $150 for the labor. In total, he paid $320 + $150 = $470. The answer is 470. Please summarize the mistakes in a short sentence for the question. At the end, please make a brief list of criteria. Make sure they are general and not specific to these questions so that others can grade the answers for other answers by following these criteria.
"""
    # Tokenize prompt
    input_ids = None
    if args.Branch_controller == "Beam_Search":
        tree_templates = None
        test_filepath = "train.jsonl"
        print(f"Loading data from {test_filepath} ...")
        list_data = load_jsonl(test_filepath)
        prompts = []
        for sample in list_data:
            prompts.append(sample["question"])
        root_prompt = prompts[args.tree_idx]
        # root_prompt = Q_reasoner
        input_ids = prompt_prepare(tokenizer, root_prompt)
    elif args.dataset is None:
        tree_templates = None
        input_ids = prompt_prepare(
            tokenizer,
            Q_reasoner,
            pad_prompt=pad_prompt,
            desired_seq_len=args.prompt_len,
        )
    else:
        if (
            args.Branch_controller == "Speculative_Decoding"
            or args.Branch_controller == "Simple_Tree"
        ):
            trees = data_loader.load_prompts(args.dataset)
        else:
            trees = data_loader.load_trees(args.dataset)
        tree_templates = trees[args.tree_idx]
        root_prompt = (
            tree_templates.prompt
            if tree_templates.prompt is not None
            else Q_reasoner
        )
        original_prompt_len = tokenizer(
            root_prompt, return_tensors="pt"
        ).input_ids
        input_ids = prompt_prepare(
            tokenizer,
            root_prompt,
            pad_prompt=pad_prompt,
            desired_seq_len=args.prompt_len,
        )
        if (
            args.Branch_controller == "Speculative_Decoding"
        ):  # update accepted len list
            data_loader.generate_accepted_len_list(
                max_gen_len=args.max_seq_len - input_ids.shape[1],
                tree=tree_templates,
            )
        # we select the first tree
        print(
            f"Information of the single tree from {args.dataset}:\n",
            f"(Token length without padding)trees[{args.tree_idx}].prompt_len={original_prompt_len.shape[1]}\n",
            f"(Token length after local model's tokenizer padding)trees[{args.tree_idx}].prompt_len={input_ids.shape[1]}\n",
            #   f"trees[{args.tree_idx}].nodes={tree_templates.nodes}\n",
            #   f"trees[{args.tree_idx}].branch_record={tree_templates.branch_record}\n",
            #   f"trees[{args.tree_idx}].prune_record={tree_templates.prune_record}\n"
        )
    perf_metrics = PerfMetrics(args.output_file)
    inference_interface(
        model,
        mode,
        tokenizer,
        input_ids,
        args.max_seq_len,
        args.max_width,
        args.max_depth,
        branch_controller=branch_controller,
        tree_template=tree_templates,
        output_file=args.output_file,
        perf_metrics=perf_metrics,
    )
    perf_metrics.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument(
        "--mode",
        choices=["node", "seq", "flatten", "tree", "node_chunk", "tree_index"],
        default="seq",
    )
    # # load tree templates
    parser.add_argument("--dataset", type=str, default=None)
    # # define the way that we do the branch controlling below.
    parser.add_argument(
        "--Branch_controller",
        choices=[
            "Simple_Tree",
            "Beam_Search",
            "Random_Tree",
            "Practical_Tree",
            "Speculative_Decoding",
        ],
        default="Simple_Tree",
    )

    parser.add_argument("--mem", choices=["paged", "unpaged"], default="paged")
    # parser.add_argument("--Sampling_method", choices=["Logits_topK", "Accu_probs_topK","User_defined" ], default="Logits_topK")
    parser.add_argument(
        "--traversal", choices=["dfs", "bfs_token", "bfs_node"], default="dfs"
    )  # how do we traverse the decoding tree
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--max_width", type=int, default=50)
    parser.add_argument(
        "--prompt_len", type=int, default=None
    )  # can pad to a wanted length
    parser.add_argument("--max_seq_len", type=int, default=500)
    parser.add_argument("--tree_idx", type=int, default=0)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--port", type=int, default=28889)
    args = parser.parse_args()
    if args.prompt_len is not None:
        if args.prompt_len <= 0:
            args.prompt_len = None
    print(
        "Generation starts with arguments:",
        ", ".join(f"{k}={v}" for k, v in vars(args).items()),
    )
    main(args)
