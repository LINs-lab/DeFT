import torch


def tree_attention_causal_masked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    attn = torch.matmul(query, key) * sm_scale + causal_mask
    attn = torch.softmax(attn, dim=2, dtype=torch.float32).to(query.dtype)
    o = torch.matmul(attn, value)
    return o


tree_attention_causal_masked_jit = torch.compile(tree_attention_causal_masked)
