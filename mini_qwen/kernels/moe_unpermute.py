"""MoE Unpermute + Weighted Reduce（M4.4 实现）。

与 moe_permute 使用同一 stable argsort，通过逆映射还原 token 顺序并加权求和。
不需要 moe_permute 返回额外的 sorted_indices，两个函数自洽。
"""
import torch


def moe_unpermute(
    permuted_output: torch.Tensor,  # [num_tokens * top_k, hidden_dim]
    topk_weights: torch.Tensor,     # [num_tokens, top_k]  float32
    topk_ids: torch.Tensor,         # [num_tokens, top_k]
    num_tokens: int,
) -> torch.Tensor:
    """加权求和还原 [num_tokens, hidden_dim]。

    重建与 moe_permute 完全相同的 stable argsort，求逆映射，
    再 gather expert 输出并按 topk_weights 加权。
    """
    T, K = topk_ids.shape
    H = permuted_output.shape[-1]

    flat = topk_ids.reshape(-1)              # [T*K]
    perm = flat.argsort(stable=True)         # 与 moe_permute 相同的 stable sort
    inv  = perm.argsort()                    # 逆映射：flat_idx[t*K+k] → perm_pos

    # 向量化 gather：perm_pos[t, k] 是 token t slot k 在 permuted_output 中的行
    perm_pos  = inv.reshape(T, K)                                        # [T, K]
    expert_h  = permuted_output[perm_pos.reshape(-1)].reshape(T, K, H)  # [T, K, H]
    out = (topk_weights.unsqueeze(-1) * expert_h).sum(dim=1)             # [T, H]
    return out.to(permuted_output.dtype)
