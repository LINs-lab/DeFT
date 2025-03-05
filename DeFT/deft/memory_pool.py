"""Memory pool."""

import logging

import torch
from typing import Optional

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int) -> None:
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.can_use_mem_size = size
        self.req_to_token = torch.empty(
            (size, max_context_len), dtype=torch.int32, device="cuda"
        )

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.can_use_mem_size:
            return None

        select_index = torch.nonzero(self.mem_state).squeeze(1)[:need_size]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        return select_index.to(torch.int32)

    def free(self, free_index: int) -> None:
        if isinstance(free_index, (int,)):
            self.can_use_mem_size += 1
        else:
            self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1

        # if self.can_use_mem_size == len(self.mem_state):
        #     print(f"ReqToTokenPool: freed all. size = {self.can_use_mem_size}.")

    def copy(self, from_req: int, to_req: int, copy_len: int) -> None:
        self.req_to_token[to_req, :copy_len] = self.req_to_token[
            from_req, :copy_len
        ]

    def clear(self) -> None:
        self.mem_state.fill_(1)
        self.can_use_mem_size = len(self.mem_state)


class TokenToKVPool:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.alloc_ct = 0

        # [size, key/value, head_num, head_dim] for each layer
        self.kv_data = [
            torch.empty(
                (size, 2, head_num, head_dim), dtype=dtype, device="cuda"
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_data[layer_id][:, 0]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_data[layer_id][:, 1]

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        select_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if select_index.shape[0] < need_size:
            return None

        self.add_refs(select_index)
        return select_index.to(torch.int32)

    def free(self, free_index: torch.Tensor) -> int:
        return self.decrease_refs(free_index)

    def used_size(self) -> int:
        return len(torch.nonzero(self.mem_state).squeeze(1))

    def available_size(self) -> int:
        return int(torch.sum(self.mem_state == 0).item())

    def add_refs(self, token_index: torch.Tensor) -> None:
        self.alloc_ct += len(token_index)
        self.mem_state[token_index] += 1

    def decrease_refs(self, token_index: torch.Tensor) -> int:
        self.alloc_ct -= len(token_index)
        self.mem_state[token_index] -= 1

        num_freed = int(torch.sum(self.mem_state[token_index] == 0).item())

        # if self.alloc_ct == 0:
        #     print(f"TokenToKVPool: freed all. size = {len(self.mem_state)}.")

        return num_freed

    def clear(self) -> None:
        self.mem_state.fill_(0)
        self.alloc_ct = 0
