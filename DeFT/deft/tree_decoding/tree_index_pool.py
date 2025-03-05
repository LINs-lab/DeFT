"""Tree Index pool."""

import logging

import torch
from typing import Optional

logger = logging.getLogger(__name__)


class TreeIndexPool:
    def __init__(self, size: int, max_context_len: int) -> None:
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.can_use_mem_size = size
        self.node_to_kv = torch.empty(
            (size, max_context_len), dtype=torch.int32, device="cuda"
        )
        self.node_to_q = torch.empty(
            (size, size), dtype=torch.int32, device="cuda"
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

    def copy(
        self, from_node: int, to_node: int, kv_len: int, q_len: int
    ) -> None:
        self.node_to_kv[to_node, :kv_len] = self.node_to_kv[from_node, :kv_len]
        self.node_to_q[to_node, :q_len] = self.node_to_q[from_node, :q_len]

    def clear(self) -> None:
        self.mem_state.fill_(1)
        self.can_use_mem_size = len(self.mem_state)

    def get_offset(self, node_id: int) -> int:
        offset = node_id * self.node_to_kv.shape[1]
        return offset
