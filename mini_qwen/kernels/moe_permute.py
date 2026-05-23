"""MoE Permute Kernel (M4.2 implementation).

Sorts tokens by expert_id to produce a contiguous memory layout for grouped GEMM.
Stable argsort ensures deterministic inverse mapping for unpermute.
"""
import torch
from typing import Tuple


def moe_permute(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_dim]
    topk_ids: torch.Tensor,       # [num_tokens, top_k]
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (permuted_hidden [num_tokens*top_k, hidden_dim], expert_offsets [num_experts+1]).

    expert_offsets[e]..expert_offsets[e+1] is the range in permuted_hidden of tokens assigned to expert e.
    """
    T, K = topk_ids.shape

    # GPU stable argsort: topk_ids is already a GPU tensor, sort directly without CPU-GPU transfer
    flat = topk_ids.reshape(-1)                              # [T*K], GPU
    perm = flat.argsort(stable=True)                         # [T*K], GPU

    # number of tokens assigned to each expert -> prefix sum
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=flat.device)
    expert_offsets[1:] = flat.bincount(minlength=num_experts).cumsum(0)

    # gather hidden_states in sorted order
    tok_idx = perm // K                                      # sorted slot -> original token index
    permuted = hidden_states[tok_idx]                        # [T*K, H]

    return permuted, expert_offsets
