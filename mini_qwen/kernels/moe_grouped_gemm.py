"""MoE Grouped GEMM Kernel (M4.3 implementation).

Computes one GEMM per expert: out[s:t] = permuted_hidden[s:t] @ W[e].T
where [s, t) is determined by expert_offsets.

Implementation:
- _moe_grouped_gemm_oracle: BF16 for-loop, used for correctness comparison
- moe_grouped_gemm: Triton kernel, dynamic grid launches only active experts
"""
import torch
import triton
import triton.language as tl


# ── for-loop oracle (correctness reference) ─────────────────────────────────

def _moe_grouped_gemm_oracle(
    permuted_hidden: torch.Tensor,   # [T*K, H]
    expert_weights: torch.Tensor,    # [E, D, H]
    expert_offsets: torch.Tensor,    # [E+1]
) -> torch.Tensor:
    E = expert_weights.shape[0]
    D = expert_weights.shape[1]
    out = torch.zeros(
        permuted_hidden.shape[0], D,
        dtype=permuted_hidden.dtype, device=permuted_hidden.device,
    )
    for e in range(E):
        s = expert_offsets[e].item()
        t = expert_offsets[e + 1].item()
        if s == t:
            continue
        # fp32 accumulation for precision
        out[s:t] = (
            permuted_hidden[s:t].float() @ expert_weights[e].float().T
        ).to(permuted_hidden.dtype)
    return out


# ── Triton kernel (performance version) ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['D', 'H'],
)
@triton.jit
def _grouped_gemm_kernel(
    X, W, Out,
    active_starts, active_ends, active_experts,
    D, H,
    stride_xr, stride_xh,
    stride_we, stride_wd, stride_wh,
    stride_or, stride_od,
    BLOCK_M: tl.constexpr,   # fixed 16, mask handles cases where n_e < 16
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)   # index of active expert
    pid_n = tl.program_id(1)   # output feature tile
    pid_m = tl.program_id(2)   # token tile (third dimension of dynamic grid)

    start = tl.load(active_starts + pid_e).to(tl.int32)
    end   = tl.load(active_ends   + pid_e).to(tl.int32)
    e     = tl.load(active_experts + pid_e).to(tl.int32)
    n_e   = end - start

    # exit early if current M tile exceeds this expert's token range
    if pid_m * BLOCK_M >= n_e:
        return

    m_off  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < n_e
    x_row  = start + m_off     # [BLOCK_M] absolute row index in permuted_hidden

    n_off  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < D

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # H is in the autotune key, determined at compile time -> loop unrolled
    for k in range((H + BLOCK_K - 1) // BLOCK_K):
        k_off  = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < H

        x_tile = tl.load(
            X + x_row[:, None] * stride_xr + k_off[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)                                       # [BLOCK_M, BLOCK_K]

        # W[e] has shape [D, H], needs to be transposed to [H, D] for X @ W.T = [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
        w_tile = tl.load(
            W + e * stride_we + n_off[:, None] * stride_wd + k_off[None, :] * stride_wh,
            mask=n_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)                                       # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    tl.store(
        Out + x_row[:, None] * stride_or + n_off[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ── public interface ──────────────────────────────────────────────────────────

_BLOCK_M = 16  # fixed M tile, aligned with Triton constexpr


def moe_grouped_gemm(
    permuted_hidden: torch.Tensor,   # [num_tokens * top_k, hidden_dim]
    expert_weights: torch.Tensor,    # [num_experts, intermediate_size, hidden_dim]
    expert_offsets: torch.Tensor,    # [num_experts + 1]
) -> torch.Tensor:
    """Grouped GEMM: computes out[s:t] = permuted_hidden[s:t] @ W[e].T for each expert.

    Automatically skips empty experts; only launches Triton programs for active experts.
    """
    if not permuted_hidden.is_cuda:
        return _moe_grouped_gemm_oracle(permuted_hidden, expert_weights, expert_offsets)

    E, D, H = expert_weights.shape
    total_slots = permuted_hidden.shape[0]
    out = torch.empty(total_slots, D, dtype=permuted_hidden.dtype, device=permuted_hidden.device)

    if total_slots == 0:
        return out

    # only launch active experts (n_e > 0), typically <= 8 during decode
    active = (expert_offsets[1:] > expert_offsets[:-1]).nonzero(as_tuple=True)[0]
    if active.shape[0] == 0:
        return out.zero_()

    active_starts = expert_offsets[active].to(torch.int32)
    active_ends   = expert_offsets[active + 1].to(torch.int32)
    active_n_e    = (active_ends - active_starts)
    max_m_tiles   = int(triton.cdiv(int(active_n_e.max().item()), _BLOCK_M))

    def grid(meta):
        return (active.shape[0], triton.cdiv(D, meta['BLOCK_N']), max_m_tiles)

    _grouped_gemm_kernel[grid](
        permuted_hidden, expert_weights, out,
        active_starts, active_ends, active.to(torch.int32),
        D, H,
        permuted_hidden.stride(0), permuted_hidden.stride(1),
        expert_weights.stride(0), expert_weights.stride(1), expert_weights.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=_BLOCK_M,
    )
    return out
