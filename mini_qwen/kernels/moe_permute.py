"""MoE Permute Kernel（M4.2 实现）。

按 expert_id 对 tokens 排序，生成连续内存布局供 grouped GEMM 使用。
stable argsort 保证 unpermute 的逆映射确定性。
"""
import torch
from typing import Tuple


def moe_permute(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_dim]
    topk_ids: torch.Tensor,       # [num_tokens, top_k]
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (permuted_hidden [num_tokens*top_k, hidden_dim], expert_offsets [num_experts+1])。

    expert_offsets[e]..expert_offsets[e+1] 是属于 expert e 的 token 在 permuted_hidden 中的范围。
    """
    T, K = topk_ids.shape

    # GPU stable argsort：topk_ids 本身是 GPU tensor，直接排序无需 CPU-GPU 搬运
    flat = topk_ids.reshape(-1)                              # [T*K]，GPU
    perm = flat.argsort(stable=True)                         # [T*K]，GPU

    # 每个 expert 被分到的 token 数量 → 前缀和
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=flat.device)
    expert_offsets[1:] = flat.bincount(minlength=num_experts).cumsum(0)

    # 按排序后的顺序 gather hidden_states
    tok_idx = perm // K                                      # 排序槽 → 原始 token index
    permuted = hidden_states[tok_idx]                        # [T*K, H]

    return permuted, expert_offsets
