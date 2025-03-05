from typing import List, Optional
import torch
from torch import nn
from torch.nn.parameter import Parameter
from deft.utils import set_weight_attrs


class QKVLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.input_size = self.hidden_size
        self.output_size = (
            self.num_heads + 2 * self.num_kv_heads
        ) * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size,  # q_proj
            self.num_kv_heads * self.head_size,  # k_proj
            self.num_kv_heads * self.head_size,  # v_proj
        ]

        self.linear = nn.Linear(self.input_size, self.output_size, bias=False)
        set_weight_attrs(
            self.linear.weight,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int:
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
        }
        offset = shard_offset_mapping.get(loaded_shard_id)
        assert offset is not None
        return offset

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int:
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        size = shard_size_mapping.get(loaded_shard_id)
        assert size is not None
        return size

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ) -> None:
        param_data = param.data
        if loaded_shard_id is not None:
            assert loaded_shard_id in ["q", "k", "v"]
            shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
            shard_size = self._get_shard_size_mapping(loaded_shard_id)
            param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MergedLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.output_size = sum(output_sizes)
        self.linear = nn.Linear(self.input_size, self.output_size, bias=False)
        set_weight_attrs(
            self.linear.weight,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int],
    ) -> None:
        param_data = param.data
        if loaded_shard_id is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
