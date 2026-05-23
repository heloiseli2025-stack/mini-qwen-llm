"""W4A16 GEMM Decode Kernel (M3 implementation, optimized).

Optimizations:
- GEMM kernel (M>1): 2D tile load instead of 8 independent scalar loads, reduces HBM cache misses
- M=1 dedicated GEMV kernel: scalar X broadcast + vectorized w_fp, full SM utilization
- dequant entirely in fp32; shifts use literal constants, unrolled at Triton compile time

# === FROZEN SIGNATURE ===
# def w4a16_gemm(
#     x:          [M, K], bf16
#     qweight:    [K // 8, N], int32 (packed int4, nibble i of qweight[j,n] = weight[8j+i,n])
#     scales:     [K // group_size, N], bf16
#     qzeros:     [K // group_size, N // 8], int32 (packed int4 along N dimension)
#     group_size: int = 128
# ) -> [M, N], bf16
"""
import torch
import triton
import triton.language as tl


# ── GEMM kernel (M > 1) ───────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64}, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K', 'group_size'],
)
@triton.jit
def _w4a16_gemm_kernel(
    X, QW, Scales, QZeros, Out,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_sg,  stride_sn,
    stride_zg,  stride_zn,
    stride_om,  stride_on,
    group_size: tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for g in range(K // group_size):
        g_start = g * group_size

        scale = tl.load(
            Scales + g * stride_sg + n_offs * stride_sn,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        z_col   = n_offs // 8
        z_shift = (n_offs % 8) * 4
        qz_row  = tl.load(
            QZeros + g * stride_zg + z_col * stride_zn,
            mask=n_mask, other=0,
        ).to(tl.int32)
        zero = ((qz_row >> z_shift) & 0xF).to(tl.float32)

        # process 16 K values at a time (two qweight rows), satisfying tl.dot K>=16 requirement
        for pair in tl.static_range(group_size // 16):
            k_base = g_start + pair * 16
            j0 = g * (group_size // 8) + pair * 2
            j1 = j0 + 1

            qw0 = tl.load(QW + j0 * stride_qwk + n_offs * stride_qwn,
                          mask=n_mask, other=0).to(tl.int32)   # [BLOCK_N]
            qw1 = tl.load(QW + j1 * stride_qwk + n_offs * stride_qwn,
                          mask=n_mask, other=0).to(tl.int32)   # [BLOCK_N]

            # build [16, BLOCK_N] w_fp: first 8 rows from qw0, last 8 rows from qw1
            k16_idx  = tl.expand_dims(tl.arange(0, 16), 1)    # [16, 1]: 0..15
            shifts   = (k16_idx % 8) * 4                        # [16, 1]: 0,4,...,28 × 2
            selector = k16_idx // 8                              # [16, 1]: 0*8 then 1*8
            qw_sel   = tl.where(selector == 0,
                                tl.expand_dims(qw0, 0),         # [1, BLOCK_N]
                                tl.expand_dims(qw1, 0))         # broadcast → [16, BLOCK_N]
            w_int_16 = (qw_sel >> shifts) & 0xF                              # [16, BLOCK_N]
            w_fp_16  = (w_int_16.to(tl.float32)
                        - tl.expand_dims(zero, 0)) * tl.expand_dims(scale, 0)  # [16, BLOCK_N]

            # X: [BLOCK_M, 16] coalesced 2D load
            k16_offs = k_base + tl.arange(0, 16)
            x_16 = tl.load(
                X + m_offs[:, None] * stride_xm + k16_offs[None, :] * stride_xk,
                mask=m_mask[:, None], other=0.0,
            ).to(tl.float32)  # [BLOCK_M, 16]

            # tl.dot: [BLOCK_M, 16] x [16, BLOCK_N] -> [BLOCK_M, BLOCK_N] (K=16 satisfies Triton requirement)
            acc += tl.dot(x_16, w_fp_16, out_dtype=tl.float32)

    out_ptrs = Out + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ── GEMV kernel (M=1 dedicated) ──────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N':  32}, num_warps=1),
        triton.Config({'BLOCK_N':  64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
    ],
    key=['N', 'K', 'group_size'],
)
@triton.jit
def _w4a16_gemv_kernel(
    X, QW, Scales, QZeros, Out,
    N, K,
    stride_xk,
    stride_qwk, stride_qwn,
    stride_sg,  stride_sn,
    stride_zg,  stride_zn,
    group_size: tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    pid    = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for g in range(K // group_size):
        g_start = g * group_size

        scale = tl.load(
            Scales + g * stride_sg + n_offs * stride_sn,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        z_col   = n_offs // 8
        z_shift = (n_offs % 8) * 4
        qz_row  = tl.load(
            QZeros + g * stride_zg + z_col * stride_zn,
            mask=n_mask, other=0,
        ).to(tl.int32)
        zero = ((qz_row >> z_shift) & 0xF).to(tl.float32)

        for j in tl.static_range(group_size // 8):
            k_base = g_start + j * 8
            j_row  = g * (group_size // 8) + j

            qw = tl.load(
                QW + j_row * stride_qwk + n_offs * stride_qwn,
                mask=n_mask, other=0,
            ).to(tl.int32)

            w_fp0 = ((qw & 0xF        ).to(tl.float32) - zero) * scale
            w_fp1 = (((qw >>  4) & 0xF).to(tl.float32) - zero) * scale
            w_fp2 = (((qw >>  8) & 0xF).to(tl.float32) - zero) * scale
            w_fp3 = (((qw >> 12) & 0xF).to(tl.float32) - zero) * scale
            w_fp4 = (((qw >> 16) & 0xF).to(tl.float32) - zero) * scale
            w_fp5 = (((qw >> 20) & 0xF).to(tl.float32) - zero) * scale
            w_fp6 = (((qw >> 24) & 0xF).to(tl.float32) - zero) * scale
            w_fp7 = (((qw >> 28) & 0xF).to(tl.float32) - zero) * scale

            # X: scalar broadcast; M=1 has only 1 row, K dimension is contiguous, all hits L1
            x0 = tl.load(X + (k_base + 0) * stride_xk).to(tl.float32)
            x1 = tl.load(X + (k_base + 1) * stride_xk).to(tl.float32)
            x2 = tl.load(X + (k_base + 2) * stride_xk).to(tl.float32)
            x3 = tl.load(X + (k_base + 3) * stride_xk).to(tl.float32)
            x4 = tl.load(X + (k_base + 4) * stride_xk).to(tl.float32)
            x5 = tl.load(X + (k_base + 5) * stride_xk).to(tl.float32)
            x6 = tl.load(X + (k_base + 6) * stride_xk).to(tl.float32)
            x7 = tl.load(X + (k_base + 7) * stride_xk).to(tl.float32)

            acc += x0 * w_fp0
            acc += x1 * w_fp1
            acc += x2 * w_fp2
            acc += x3 * w_fp3
            acc += x4 * w_fp4
            acc += x5 * w_fp5
            acc += x6 * w_fp6
            acc += x7 * w_fp7

    tl.store(Out + n_offs, acc.to(tl.bfloat16), mask=n_mask)


# ── public interface ──────────────────────────────────────────────────────────

def w4a16_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """W4A16 GEMM: x @ dequant(qweight).

    Args:
        x:          [M, K] bf16
        qweight:    [K//8, N] int32
        scales:     [K//group_size, N] bf16
        qzeros:     [K//group_size, N//8] int32
        group_size: default 128

    Returns:
        [M, N] bf16
    """
    assert x.is_cuda and qweight.is_cuda
    assert x.dtype == torch.bfloat16
    M, K = x.shape
    N    = qweight.shape[1]
    assert K % group_size == 0
    assert N % 8 == 0

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    if M == 1:
        _w4a16_gemv_kernel[lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)](
            x, qweight, scales, qzeros, out,
            N, K,
            x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0),  scales.stride(1),
            qzeros.stride(0),  qzeros.stride(1),
            group_size=group_size,
        )
    else:
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        _w4a16_gemm_kernel[grid](
            x, qweight, scales, qzeros, out,
            M, N, K,
            x.stride(0),       x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0),  scales.stride(1),
            qzeros.stride(0),  qzeros.stride(1),
            out.stride(0),     out.stride(1),
            group_size=group_size,
        )

    return out
