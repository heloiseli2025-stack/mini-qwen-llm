"""MoE Top-K Router (M4.1 implementation)."""
import torch
import torch.nn.functional as F
from typing import Tuple


def moe_router(
    hidden_states: torch.Tensor,   # [num_tokens, hidden_dim]
    router_weight: torch.Tensor,   # [num_experts, hidden_dim]
    top_k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k routing.

    Returns:
        topk_ids:     [num_tokens, top_k] int64
        topk_weights: [num_tokens, top_k] float32
    """
    # fp32 matmul + fp32 softmax: bf16 logit errors are amplified by softmax and affect weight values
    logits = F.linear(hidden_states.float(), router_weight.float())  # [T, E] fp32
    scores = F.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    return topk_ids, topk_weights
