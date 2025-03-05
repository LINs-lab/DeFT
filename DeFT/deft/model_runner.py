import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Tuple
import time
from enum import Enum, auto

import numpy as np
import torch
import torch.distributed
from deft.memory_pool import ReqToTokenPool, TokenToKVPool
from deft.model_config import ModelConfig
from deft.utils import get_available_gpu_memory, set_default_torch_dtype
from deft.logger import create_logger

from deft.tree_decoding.tree_cache import (
    TreeMetadata,
    TreeCache,
    KVCacheUpdater,
    register_tree_metadata,
    unregister_tree_metadata,
    register_tree_cache,
)
from deft.tree_decoding.tree_index_pool import TreeIndexPool
from deft.tree_decoding.timer import GlobalTimer

logger = create_logger(__name__)


class ForwardMode(Enum):
    PREFILL = auto()
    EXTEND = auto()
    DECODE = auto()
    TREE_DECODE = auto()
    TREE_DECODE_NODE = auto()
    TREE_DECODE_FLATTEN = auto()
    TREE_DECODE_INDEX_NODE = auto()
    UNPAGED_FD = auto()
    UNPAGED_MEDUSA = auto()
    UNPAGED_DEFT_NODE = auto()
    UNPAGED_DEFT_FLATTEN = auto()


@lru_cache()
def import_model_classes() -> dict[str, type]:
    model_arch_name_to_cls = {}
    for module_path in (Path(__file__).parent / "models").glob("*.py"):
        module = importlib.import_module(f"deft.models.{module_path.stem}")
        if hasattr(module, "EntryClass"):
            model_arch_name_to_cls[module.EntryClass.__name__] = (
                module.EntryClass
            )
    return model_arch_name_to_cls


def get_model_cls_by_arch_name(model_arch_names: List[str]) -> type:
    model_arch_name_to_cls = import_model_classes()

    model_class = None
    for arch in model_arch_names:
        if arch in model_arch_name_to_cls:
            model_class = model_arch_name_to_cls[arch]
            break
    else:
        raise ValueError(
            f"Unsupported architectures: {arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_class


@dataclass
class InputMetadata:
    # model_runner: "ModelRunner"
    forward_mode: ForwardMode
    batch_size: int
    total_num_tokens: int
    max_seq_len: int
    req_pool_indices: torch.Tensor
    start_loc: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: Optional[ReqToTokenPool]
    token_to_kv_pool: Optional[TokenToKVPool]

    kv_updater: KVCacheUpdater

    other_kv_index: Optional[int] = None
    return_logprob: bool = False

    # for DeFT
    # tree_metadata: Optional[TreeMetadata] = None

    @classmethod
    def create(
        cls,
        # model_runner: "ModelRunner",
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool: Optional[TokenToKVPool],
        forward_mode: ForwardMode,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        position_ids_offsets: torch.Tensor,
        kv_updater: KVCacheUpdater,
        return_logprob: bool = False,
    ) -> "InputMetadata":
        batch_size = len(req_pool_indices)
        start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
        total_num_tokens = int(torch.sum(seq_lens).item())
        max_seq_len = int(torch.max(seq_lens))
        other_kv_index = None
        if forward_mode == ForwardMode.DECODE:
            positions = ((seq_lens - 1) + position_ids_offsets).to(torch.int64)
            if req_to_token_pool is not None:
                other_kv_index = int(
                    req_to_token_pool.req_to_token[
                        req_pool_indices[0], seq_lens[0] - 1
                    ].item()
                )
        else:
            seq_lens_np = seq_lens.cpu().numpy()
            prefix_lens_np = prefix_lens.cpu().numpy()
            position_ids_offsets_np = position_ids_offsets.cpu().numpy()
            positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(
                            prefix_lens_np[i] + position_ids_offsets_np[i],
                            seq_lens_np[i] + position_ids_offsets_np[i],
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                ),
                device="cuda",
            )

        ret = cls(
            # model_runner=model_runner,
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            start_loc=start_loc,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            positions=positions,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            kv_updater=kv_updater,
            return_logprob=return_logprob,
            other_kv_index=other_kv_index,
        )

        return ret

    @classmethod
    def from_tree(
        cls,
        # model_runner: "ModelRunner",
        tree: TreeCache,
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool: Optional[TokenToKVPool],
        forward_mode: ForwardMode,
        positions: torch.Tensor,
        kv_updater: KVCacheUpdater,
        # tree_metadata: Optional[TreeMetadata],
        return_logprob: bool = False,
    ) -> "InputMetadata":
        batch_size = positions.shape[0]
        seq_lens = positions + 1
        start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
        max_seq_len = int(torch.max(seq_lens))
        total_num_tokens = int(torch.sum(seq_lens))
        # req_pool_indices = torch.arange(batch_size, device="cuda")
        req_pool_indices = torch.tensor(
            [
                v
                for _, v in sorted(tree.leaf_to_req.items(), key=lambda x: x[0])
            ],
            dtype=torch.int32,
            device="cuda",
        )
        other_kv_index = None
        if forward_mode == ForwardMode.DECODE:
            assert req_to_token_pool is not None
            # assert token_to_kv_pool is not None
            # for leaf in tree.leaves.values():
            #     kv_indices = leaf.kv_indices
            #     node = leaf
            #     while node.parent is not None:
            #         node = node.parent
            #         kv_indices = node.kv_indices + kv_indices
            #     req_to_token_pool.req_to_token[
            #         tree_metadata.leaf_to_q[leaf.id],
            #         : seq_lens[tree_metadata.leaf_to_q[leaf.id]],
            #     ] = torch.tensor(kv_indices, dtype=torch.int32, device="cuda")

            other_kv_index = int(
                req_to_token_pool.req_to_token[
                    req_pool_indices[0], seq_lens[0] - 1
                ].item()
            )

        ret = cls(
            # model_runner=model_runner,
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            start_loc=start_loc,
            seq_lens=seq_lens,
            prefix_lens=torch.zeros_like(seq_lens),
            positions=positions,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            kv_updater=kv_updater,
            return_logprob=return_logprob,
            other_kv_index=other_kv_index,
            # tree_metadata=tree_metadata,
        )
        # ret.init_flashinfer_args(1)

        return ret


class ModelRunner:
    def __init__(
        self,
        model_config: "ModelConfig",
        mem_fraction_static: float = 0.9,
        load_format: str = "auto",
        trust_remote_code: bool = True,
        use_paged_memory: bool = True,
        use_tree_index: bool = False,
    ):
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.load_format = load_format
        self.trust_remote_code = trust_remote_code
        self.decode_time = 0.0
        self.use_paged_memory = use_paged_memory
        self.use_tree_index = use_tree_index

        # initialize_model_parallel(tensor_model_parallel_size=self.tp_size)

        total_gpu_memory = get_available_gpu_memory(0) * (1 << 30)
        self.load_model()
        self.init_memory_pool(total_gpu_memory)
        logger.info(
            f"ModelRunner: max_total_num_token: {self.max_total_num_token}"
        )

    def load_model(self) -> None:
        """See also vllm/model_executor/model_loader.py::get_model"""
        # Select model class
        architectures = getattr(
            self.model_config.hf_config, "architectures", []
        )
        model_class = get_model_cls_by_arch_name(architectures)
        logger.info("load weight begin.")

        # Load weights
        with set_default_torch_dtype(torch.float16):
            with torch.device("cuda"):
                # hf_quant_config = getattr(
                #     self.model_config.hf_config, "quantization_config", None
                # )
                # if hf_quant_config is not None:
                #     quant_config_class = QUANTIONCONFIG_MAPPING.get(
                #         hf_quant_config["quant_method"]
                #     )
                #     if quant_config_class is None:
                #         raise ValueError(
                #             f"Unsupported quantization method: {hf_quant_config['quant_method']}"
                #         )
                #     quant_config = quant_config_class.from_config(hf_quant_config)
                #     logger.info(f"quant_config: {quant_config}")
                #     linear_method = quant_config.get_linear_method()
                model = model_class(
                    config=self.model_config.hf_config,
                )
            model.load_weights(
                self.model_config.path,
                cache_dir=None,
                load_format=self.load_format,
                revision=None,
            )
        self.model = model.eval()

        logger.info("load weight end.")

    def profile_max_num_token(self, total_gpu_memory: float) -> int:
        available_gpu_memory = get_available_gpu_memory(0) * (1 << 30)
        head_dim = (
            self.model_config.hidden_size
            // self.model_config.num_attention_heads
        )
        head_num = self.model_config.num_key_value_heads
        cell_size = (
            head_num * head_dim * self.model_config.num_hidden_layers * 2 * 2
        )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory // cell_size)
        return max_num_token

    def init_memory_pool(self, total_gpu_memory: float) -> None:
        self.max_total_num_token = self.profile_max_num_token(total_gpu_memory)
        if self.max_total_num_token <= 0:
            raise RuntimeError(
                "Not enought memory. "
                "Please try to increase --mem-fraction-static."
            )
        self.req_to_token_pool: Optional[ReqToTokenPool] = None
        self.token_to_kv_pool: Optional[TokenToKVPool] = None
        self.tree_index_pool: Optional[TreeIndexPool] = None
        self.max_requests = int(
            self.max_total_num_token / self.model_config.context_len * 256
        )
        if self.use_paged_memory:
            self.req_to_token_pool = ReqToTokenPool(
                self.max_requests,
                self.model_config.context_len + 8,
            )
            self.token_to_kv_pool = TokenToKVPool(
                self.max_total_num_token,
                dtype=torch.float16,
                head_num=self.model_config.num_key_value_heads,
                head_dim=self.model_config.hidden_size
                // self.model_config.num_attention_heads,
                layer_num=self.model_config.num_hidden_layers,
            )
        if self.use_tree_index:
            self.tree_index_pool = TreeIndexPool(
                self.max_requests, self.model_config.context_len + 8
            )

        self.tree = TreeCache(
            torch.float16,
            self.model_config.num_key_value_heads,
            self.model_config.hidden_size
            // self.model_config.num_attention_heads,
            self.model_config.num_hidden_layers,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_index_pool,
            self.use_paged_memory,
            self.use_tree_index,
        )
        register_tree_cache(self.tree)

    @torch.inference_mode()
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        position_ids_offsets: torch.Tensor,
        kv_updater: KVCacheUpdater,
        return_logprob: bool,
    ) -> torch.Tensor:
        # NOTE(jinwei): maybe we should include tree metadata here.
        input_metadata = InputMetadata.create(
            self.req_to_token_pool,
            self.token_to_kv_pool,
            forward_mode=ForwardMode.PREFILL,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            position_ids_offsets=position_ids_offsets,
            kv_updater=kv_updater,
            return_logprob=return_logprob,
        )
        return self.model.forward(
            input_ids, input_metadata.positions, input_metadata
        )

    @torch.inference_mode()
    def forward_tree_decode(
        self,
        forward_mode: ForwardMode,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_updater: KVCacheUpdater,
        return_logprob: bool,
        tree_metadata: Optional[TreeMetadata],
    ) -> Tuple[torch.Tensor, float]:
        # NOTE(jinwei): maybe we should include tree metadata here.
        GlobalTimer.start("input_metadata")
        input_metadata = InputMetadata.from_tree(
            tree=self.tree,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            forward_mode=forward_mode,
            positions=positions,
            kv_updater=kv_updater,
            # tree_metadata=tree_metadata,
            return_logprob=return_logprob,
        )
        if tree_metadata is not None:
            register_tree_metadata(tree_metadata)
        # input_metadata = weakref.proxy(input_metadata)
        GlobalTimer.stop("input_metadata")
        GlobalTimer.stop("prepare")
        torch.cuda.synchronize()
        start = time.time()
        # GlobalTimer.start("forward")
        result = self.model.forward(input_ids, positions, input_metadata)
        # GlobalTimer.stop("forward")
        torch.cuda.synchronize()
        t = time.time() - start
        self.decode_time += t
        unregister_tree_metadata()
        return result, t
