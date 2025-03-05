# Adapted from
# https://github.com/vllm-project/vllm/blob/671af2b1c0b3ed6d856d37c21a561cc429a10701/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

from typing import Any, Dict, Optional, Tuple

import torch
from deft.layers.logits_processor import LogitsProcessor
from deft.model_runner import InputMetadata
from torch import nn
from transformers import LlamaConfig
from deft.layers.activation import SiluAndMul
from deft.layers.layernorm import RMSNorm
from deft.layers.rotary_embedding import get_rope
from deft.layers.linear import QKVLinear, MergedLinear
from deft.layers.attention.deft_attention import DeFTAttention

from deft.utils import hf_model_weights_iterator, default_weight_loader
from deft.logger import create_logger

logger = create_logger(__name__)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(
            hidden_size,
            [intermediate_size] * 2,
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: int = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dtype=torch.float32,
        )
        self.attn = DeFTAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(
            config, "max_position_embeddings", 8192
        )
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        assert residual is not None
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        skip_embed: bool = False,
    ) -> torch.Tensor:
        if not skip_embed:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_ids
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        skip_embed: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, input_metadata, skip_embed
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, input_metadata
        )

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ) -> None:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj.linear", "q_proj", "q"),
            ("qkv_proj.linear", "k_proj", "k"),
            ("qkv_proj.linear", "v_proj", "v"),
            ("gate_up_proj.linear", "gate_proj", 0),
            ("gate_up_proj.linear", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        # logger.info(f"param keys: {params_dict.keys()}")
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if (
                "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
            ):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                assert hasattr(param, "weight_loader")
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)


EntryClass = LlamaForCausalLM
