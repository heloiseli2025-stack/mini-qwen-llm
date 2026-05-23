"""MoE Unpermute + Weighted Reduce (M4.4 implementation).

Uses the same stable argsort as moe_permute to restore token order via inverse mapping and perform weighted summation.
Does not require moe_permute to return additional sorted_indices; the two functions are self-consistent.
"""
import torch


def moe_unpermute(
    permuted_output: torch.Tensor,  # [num_tokens * top_k, hidden_dim]
    topk_weights: torch.Tensor,     # [num_tokens, top_k]  float32
    topk_ids: torch.Tensor,         # [num_tokens, top_k]
    num_tokens: int,
) -> torch.Tensor:
    """Weighted reduce to restore [num_tokens, hidden_dim].

    Reconstructs the exact same stable argsort as moe_permute, computes the inverse mapping,
    then gathers expert outputs and weights them by topk_weights.
    """
    T, K = topk_ids.shape
    H = permuted_output.shape[-1]

    flat = topk_ids.reshape(-1)              # [T*K]
    perm = flat.argsort(stable=True)         # same stable sort as moe_permute
    inv  = perm.argsort()                    # inverse mapping: flat_idx[t*K+k] -> perm_pos

    # vectorized gather: perm_pos[t, k] is the row in permuted_output for token t slot k
    perm_pos  = inv.reshape(T, K)                                        # [T, K]
    expert_h  = permuted_output[perm_pos.reshape(-1)].reshape(T, K, H)  # [T, K, H]
    out = (topk_weights.unsqueeze(-1) * expert_h).sum(dim=1)             # [T, H]
    return out.to(permuted_output.dtype)
