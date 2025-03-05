from typing import Optional
import torch
from deft.hf_transformers_utils import get_config, get_context_length
from typing import cast

# adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class ModelConfig:
    def __init__(
        self,
        path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> None:
        self.path = path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.hf_config = get_config(self.path, trust_remote_code, revision)
        config_dtype = getattr(self.hf_config, "torch_dtype", "bfloat16")
        if isinstance(config_dtype, str):
            if config_dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {config_dtype}")
            self.dtype = _STR_DTYPE_TO_TORCH_DTYPE[config_dtype]
        elif isinstance(config_dtype, torch.dtype):
            self.dtype = config_dtype
        else:
            raise ValueError(f"Unknown dtype: {config_dtype}")

        if context_length is not None:
            self.context_len = context_length
        else:
            self.context_len = get_context_length(self.hf_config)

        # Unify the config keys for hf_config
        self.head_dim = cast(
            int,
            (self.hf_config.hidden_size // self.hf_config.num_attention_heads),
        )
        self.num_attention_heads = cast(int, self.hf_config.num_attention_heads)
        num_key_value_heads = cast(
            Optional[int], getattr(self.hf_config, "num_key_value_heads", None)
        )
        if num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.hidden_size = cast(int, self.hf_config.hidden_size)
        self.num_hidden_layers = cast(int, self.hf_config.num_hidden_layers)
        self.vocab_size = cast(int, self.hf_config.vocab_size)
