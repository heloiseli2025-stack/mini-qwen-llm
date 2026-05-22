"""MoE Grouped GEMM Kernel（M4.3 实现）。

为每个 expert 计算一个 GEMM：out[s:t] = permuted_hidden[s:t] @ W[e].T
其中 [s, t) 由 expert_offsets 确定。

实现：
- _moe_grouped_gemm_oracle：BF16 for-loop，用于正确性对比
- moe_grouped_gemm：Triton kernel，动态 grid 只启动 active experts
"""
import torch
import triton
import triton.language as tl


# ── for-loop oracle（正确性参考）────────────────────────────────────────────────

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


# ── Triton kernel（性能版）───────────────────────────────────────────────────

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
    BLOCK_M: tl.constexpr,   # 固定 16，通过 mask 处理 n_e < 16 的情况
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)   # 第几个 active expert
    pid_n = tl.program_id(1)   # output feature tile
    pid_m = tl.program_id(2)   # token tile（动态 grid 的第三维）

    start = tl.load(active_starts + pid_e).to(tl.int32)
    end   = tl.load(active_ends   + pid_e).to(tl.int32)
    e     = tl.load(active_experts + pid_e).to(tl.int32)
    n_e   = end - start

    # 当前 M tile 超出该 expert 的 token 范围则直接退出
    if pid_m * BLOCK_M >= n_e:
        return

    m_off  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < n_e
    x_row  = start + m_off     # [BLOCK_M] 在 permuted_hidden 里的绝对行索引

    n_off  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < D

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # H 在 autotune key 里，编译时确定 → 循环展开
    for k in range((H + BLOCK_K - 1) // BLOCK_K):
        k_off  = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < H

        x_tile = tl.load(
            X + x_row[:, None] * stride_xr + k_off[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)                                       # [BLOCK_M, BLOCK_K]

        # W[e] 形状 [D, H]，需要转置成 [H, D] 做 X @ W.T = [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N]
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


# ── 公开接口 ──────────────────────────────────────────────────────────────────

_BLOCK_M = 16  # 固定 M tile，与 Triton constexpr 对齐


def moe_grouped_gemm(
    permuted_hidden: torch.Tensor,   # [num_tokens * top_k, hidden_dim]
    expert_weights: torch.Tensor,    # [num_experts, intermediate_size, hidden_dim]
    expert_offsets: torch.Tensor,    # [num_experts + 1]
) -> torch.Tensor:
    """Grouped GEMM：为每个 expert 计算 out[s:t] = permuted_hidden[s:t] @ W[e].T。

    自动跳过空 expert；只对 active expert 启动 Triton program。
    """
    if not permuted_hidden.is_cuda:
        return _moe_grouped_gemm_oracle(permuted_hidden, expert_weights, expert_offsets)

    E, D, H = expert_weights.shape
    total_slots = permuted_hidden.shape[0]
    out = torch.empty(total_slots, D, dtype=permuted_hidden.dtype, device=permuted_hidden.device)

    if total_slots == 0:
        return out

    # 只启动 active expert（n_e > 0），decode 时通常 ≤ 8 个
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
