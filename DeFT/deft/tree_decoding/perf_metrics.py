from typing import Optional, List
import json
from tabulate import tabulate


class PerfMetrics:
    e2e_latency: float = 0
    decode_latency: float = 0
    attention_latency: float = 0
    prompt_len: int = 0
    generated_len: int = 0
    TTFT: float = 0
    TPOT: float = 0
    KV_IO: float = 0
    QO_IO: float = 0
    Mask_IO: float = 0
    QK_IO: float = 0
    QK_scale_IO: float = 0
    QK_scale_masked_IO: float = 0
    SoftMax_IO: float = 0

    def __init__(self, output_file: Optional[str]):
        self.output_file = output_file
        self.iter_time: List[float] = []
        self.prepare_per_iter: List[float] = []
        self.forward_per_iter: List[float] = []
        self.branch_per_iter: List[float] = []
        self.attn_mem_per_iter: List[float] = []
        self.attn_comp_per_iter: List[float] = []
        self.traversal_per_iter: List[float] = []
        self.alloc_per_iter: List[float] = []
        self.positions_per_iter: List[float] = []
        self.tree_metadata_per_iter: List[float] = []
        self.input_metadata_per_iter: List[float] = []

    def update(
        self,
        iter_time: float,
        prepare: float,
        forward: float,
        branch: float,
        attn_mem: float,
        attn_comp: float,
        traversal: float,
        alloc: float,
        positions: float,
        tree_metadata: float,
        input_metadata: float,
    ) -> None:
        self.iter_time.append(iter_time)
        self.prepare_per_iter.append(prepare)
        self.forward_per_iter.append(forward)
        self.branch_per_iter.append(branch)
        self.attn_mem_per_iter.append(attn_mem)
        self.attn_comp_per_iter.append(attn_comp)
        self.traversal_per_iter.append(traversal)
        self.alloc_per_iter.append(alloc)
        self.positions_per_iter.append(positions)
        self.tree_metadata_per_iter.append(tree_metadata)
        self.input_metadata_per_iter.append(input_metadata)

    def dump(self) -> None:
        metrics = {
            "e2e_latency": PerfMetrics.e2e_latency,
            "decode_latency": self.decode_latency,
            "attention_latency": self.attention_latency,
            "prompt_len": PerfMetrics.prompt_len,
            "generated_len": PerfMetrics.generated_len,
            "TTFT": self.TTFT,
            "TPOT": self.TPOT,
            "KV_IO": PerfMetrics.KV_IO,
            "QO_IO": PerfMetrics.QO_IO,
            "Mask_IO": PerfMetrics.Mask_IO,
            "QK_IO": PerfMetrics.QK_IO,
            "QK_scale_IO": PerfMetrics.QK_scale_IO,
            "QK_scale_masked_IO": PerfMetrics.QK_scale_masked_IO,
            "SoftMax_IO": PerfMetrics.SoftMax_IO,
            "iter_time": self.iter_time,
            "prepare_per_iter": self.prepare_per_iter,
            "forward_per_iter": self.forward_per_iter,
            "branch_per_iter": self.branch_per_iter,
            "attn_mem_per_iter": self.attn_mem_per_iter,
            "attn_comp_per_iter": self.attn_comp_per_iter,
            "traversal_per_iter": self.traversal_per_iter,
            "alloc_per_iter": self.alloc_per_iter,
            "positions_per_iter": self.positions_per_iter,
            "tree_metadata_per_iter": self.tree_metadata_per_iter,
            "input_metadata_per_iter": self.input_metadata_per_iter,
        }
        if self.output_file is not None:
            with open(self.output_file, "w") as f:
                json.dump(metrics, f)

    @staticmethod
    def update_e2e_latency(e2e_latency: float) -> None:
        PerfMetrics.e2e_latency = e2e_latency

    def update_decode_latency(self) -> float:
        self.decode_latency = sum(self.forward_per_iter)
        return self.decode_latency

    def update_attention_latency(self) -> float:
        self.attention_latency = sum(self.attn_mem_per_iter) + sum(
            self.attn_comp_per_iter
        )
        return self.attention_latency

    def get_attention_mem_latency(self) -> float:
        """Get Attention Memory Management Latency"""
        return sum(self.attn_mem_per_iter)

    def get_attention_comp_latency(self) -> float:
        """Get Attention Computation Latency"""
        return sum(self.attn_comp_per_iter)

    @staticmethod
    def update_KV_IO(kv_len: int, hidden_size: int) -> None:
        PerfMetrics.KV_IO += kv_len * hidden_size * 4  # bytes

    @staticmethod
    def update_Mask_IO(kv_len: int) -> None:
        PerfMetrics.Mask_IO += kv_len * 8  # bytes

    @staticmethod
    def update_Causal_Tree_Attn_IO(
        query_len: int, kv_len: int, head_num: int, head_size: int
    ) -> None:
        PerfMetrics.QO_IO += (
            query_len * head_num * head_size * 2
        )  # read Q(fp16)
        PerfMetrics.KV_IO += kv_len * head_num * head_size * 2  # read K(fp16)
        PerfMetrics.QK_IO += query_len * kv_len * head_num * 2  # write QK(fp16)

        PerfMetrics.QK_IO += query_len * kv_len * head_num * 2  # read QK(fp16)
        PerfMetrics.QK_scale_IO += (
            query_len * kv_len * head_num * 2
        )  # write QK/d(fp16)

        PerfMetrics.QK_scale_IO += (
            query_len * kv_len * head_num * 2
        )  # read QK/d(fp16)
        PerfMetrics.Mask_IO += query_len * kv_len * 2  # read mask(fp16)
        PerfMetrics.QK_scale_masked_IO += (
            query_len * kv_len * head_num * 2
        )  # write QK/d + m(fp16)

        PerfMetrics.QK_scale_masked_IO += (
            query_len * kv_len * head_num * 2
        )  # read QK/d + m(fp16)
        PerfMetrics.SoftMax_IO += (
            query_len * kv_len * head_num * 4
        )  # write softmax(QK/d + m)(fp32)
        PerfMetrics.SoftMax_IO += (
            query_len * kv_len * head_num * 2
        )  # write softmax(QK/d + m)(fp16)

        PerfMetrics.SoftMax_IO += (
            query_len * kv_len * head_num * 2
        )  # read softmax(QK/d + m)(fp16)
        PerfMetrics.KV_IO += kv_len * head_num * head_size * 2  # read V(fp16)
        PerfMetrics.QO_IO += (
            query_len * head_num * head_size * 2
        )  # write O(fp16)

    def print_latency(
        self,
        prompt_len: Optional[int] = None,
        generated_len: Optional[int] = None,
        ttft: Optional[float] = None,
    ) -> None:
        """Prints latency metrics and optionally includes token statistics."""

        data = [
            [
                "E2E Latency (Include Overheads)",
                f"{PerfMetrics.e2e_latency:.2f} ms",
            ],
            [
                "Decode Latency (Optimized E2E)",
                f"{self.update_decode_latency():.2f} ms",
            ],
            [
                "Total Attention Latency",
                f"{self.update_attention_latency():.2f} ms",
            ],
            [
                "  - Memory Management Latency",
                f"{self.get_attention_mem_latency():.2f} ms",
            ],
            [
                "  - Attention Computation Latency",
                f"{self.get_attention_comp_latency():.2f} ms",
            ],
        ]

        # Add optional metrics only if they are provided
        if prompt_len is not None:
            data.append(["Prompt Token Number", f"{prompt_len} tokens"])
            PerfMetrics.prompt_len = prompt_len
        if ttft is not None:
            data.append(["Time To First Token (TTFT)", f"{ttft:.2f} ms"])
            PerfMetrics.TTFT = ttft
        if generated_len is not None:
            tpot = self.decode_latency / generated_len
            data.append(["Generated Token Number", f"{generated_len} tokens"])
            data.append(
                ["Time Per Output Token (TPOT)", f"{tpot:.2f} ms/token"]
            )
            PerfMetrics.generated_len = generated_len
            PerfMetrics.TPOT = tpot

        print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))

        print(
            "\nNote: E2E Latency includes unnecessary overheads such as profiling, timers, and branching, etc."
        )
        print(
            "In DeFT paper, we adopt Decode Latency as our optimized E2E latency, which includes attention and MLP latency."
        )
