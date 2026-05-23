"""Fused QKV Projection + QK-Norm + RoPE Kernel (M2 implementation).

Implementation strategy:
  1. QKV projection: 3x torch.mm (cuBLAS), not in Triton (cuBLAS is faster for large GEMMs)
  2. QK-Norm + RoPE: 1 Triton kernel, Q and K processed together in the same grid
  3. V: used directly from GEMM output, no norm / RoPE

Grid: (B*S, H_Q + H_KV)
  pid_head 0..H_Q-1          -> process Q head (is_q = True)
  pid_head H_Q..H_Q+H_KV-1  -> process K head (is_q = False)

HBM reads/writes (per Q/K):
  unfused: GEMM write + norm read + norm write + rope read + rope write = 5 trips
  fused  : GEMM write + kernel read + kernel write                      = 3 trips

# === FROZEN SIGNATURE === (frozen after M2 is complete)
# def fused_qkv_rope(
#     x:        torch.Tensor,  # [batch, seq_len, hidden_size], bf16
#     w_q:      torch.Tensor,  # [num_q_heads * head_dim, hidden_size], bf16
#     w_k:      torch.Tensor,  # [num_kv_heads * head_dim, hidden_size], bf16
#     w_v:      torch.Tensor,  # [num_kv_heads * head_dim, hidden_size], bf16
#     q_norm_w: torch.Tensor,  # [head_dim], bf16
#     k_norm_w: torch.Tensor,  # [head_dim], bf16
#     cos:      torch.Tensor,  # [seq_len, head_dim], bf16
#     sin:      torch.Tensor,  # [seq_len, head_dim], bf16
# ) -> Tuple[Tensor, Tensor, Tensor]  # q [B,S,H_q,D], k [B,S,H_kv,D], v [B,S,H_kv,D], bf16
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    # ── inputs ────────────────────────────────────────────────────
    Q,          # [total_tokens, H_Q, D]  raw GEMM output, bf16
    K,          # [total_tokens, H_KV, D] raw GEMM output, bf16
    # ── outputs ───────────────────────────────────────────────────
    Q_out,      # [total_tokens, H_Q, D]  bf16
    K_out,      # [total_tokens, H_KV, D] bf16
    # ── norm weights ──────────────────────────────────────────────
    q_norm_w,   # [D] bf16
    k_norm_w,   # [D] bf16
    # ── RoPE tables ───────────────────────────────────────────────
    cos,        # [S, D] bf16
    sin,        # [S, D] bf16
    # ── runtime scalars ───────────────────────────────────────────
    S,          # seq_len, used for seq_pos = pid_tok % S
    H_Q,        # number of Q heads, used to distinguish Q/K branches
    stride_qt,  # token stride of Q = H_Q * D
    stride_kt,  # token stride of K = H_KV * D
    # ── compile-time constants ────────────────────────────────────
    D:      tl.constexpr,   # head_dim (128)
    HALF_D: tl.constexpr,   # head_dim // 2 (64)
    EPS:    tl.constexpr,   # RMSNorm eps (1e-6)
):
    pid_tok  = tl.program_id(0)   # 0 .. B*S - 1
    pid_head = tl.program_id(1)   # 0 .. H_Q+H_KV - 1

    is_q = pid_head < H_Q

    # K head index: Q branch uses 0 (placeholder, not actually accessed)
    k_head = tl.where(is_q, 0, pid_head - H_Q)

    d0 = tl.arange(0, HALF_D)   # [HALF_D]

    # ── load inputs (split into two halves, force fp32 accumulation) ─────────
    if is_q:
        base = Q + pid_tok * stride_qt + pid_head * D
        x1 = tl.load(base + d0        ).to(tl.float32)
        x2 = tl.load(base + HALF_D + d0).to(tl.float32)
        w1 = tl.load(q_norm_w + d0        ).to(tl.float32)
        w2 = tl.load(q_norm_w + HALF_D + d0).to(tl.float32)
    else:
        base = K + pid_tok * stride_kt + k_head * D
        x1 = tl.load(base + d0        ).to(tl.float32)
        x2 = tl.load(base + HALF_D + d0).to(tl.float32)
        w1 = tl.load(k_norm_w + d0        ).to(tl.float32)
        w2 = tl.load(k_norm_w + HALF_D + d0).to(tl.float32)

    # ── RMSNorm (per-head, fp32 accumulation to prevent bf16 precision loss) ──
    rms = tl.rsqrt((tl.sum(x1 * x1) + tl.sum(x2 * x2)) / D + EPS)
    x1_n = x1 * rms * w1
    x2_n = x2 * rms * w2

    # ── RoPE ─────────────────────────────────────────────────────
    # rotate_half([x1, x2]) = [-x2, x1]
    # out[:HALF_D] = x1 * cos[:HALF_D] - x2 * sin[:HALF_D]
    # out[HALF_D:] = x2 * cos[HALF_D:] + x1 * sin[HALF_D:]
    seq_pos = pid_tok % S
    cos_base = cos + seq_pos * D
    sin_base = sin + seq_pos * D
    cos1 = tl.load(cos_base + d0        ).to(tl.float32)
    cos2 = tl.load(cos_base + HALF_D + d0).to(tl.float32)
    sin1 = tl.load(sin_base + d0        ).to(tl.float32)
    sin2 = tl.load(sin_base + HALF_D + d0).to(tl.float32)

    out1 = (x1_n * cos1 - x2_n * sin1).to(tl.bfloat16)
    out2 = (x2_n * cos2 + x1_n * sin2).to(tl.bfloat16)

    # ── write back ────────────────────────────────────────────────
    if is_q:
        o_base = Q_out + pid_tok * stride_qt + pid_head * D
    else:
        o_base = K_out + pid_tok * stride_kt + k_head * D

    tl.store(o_base + d0        , out1)
    tl.store(o_base + HALF_D + d0, out2)


def fused_qkv_rope(
    x:        torch.Tensor,   # [B, S, hidden_size], bf16
    w_q:      torch.Tensor,   # [H_Q * D, hidden_size], bf16
    w_k:      torch.Tensor,   # [H_KV * D, hidden_size], bf16
    w_v:      torch.Tensor,   # [H_KV * D, hidden_size], bf16
    q_norm_w: torch.Tensor,   # [D], bf16
    k_norm_w: torch.Tensor,   # [D], bf16
    cos:      torch.Tensor,   # [S, D], bf16
    sin:      torch.Tensor,   # [S, D], bf16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV projection + QK-Norm + RoPE.

    Returns (q, k, v), all with shape [B, S, H_heads, D], dtype bf16.
    - q: [B, S, H_Q, D]
    - k: [B, S, H_KV, D]
    - v: [B, S, H_KV, D]  directly from GEMM output, no norm/rope
    """
    B, S, H = x.shape
    D    = q_norm_w.shape[0]
    H_Q  = w_q.shape[0] // D
    H_KV = w_k.shape[0] // D

    assert D % 2 == 0, "head_dim must be even (RoPE rotate_half)"
    HALF_D = D // 2

    x_2d = x.reshape(B * S, H)

    # ① three cuBLAS GEMMs (one kernel launch each)
    q = (x_2d @ w_q.T).view(B * S, H_Q,  D)
    k = (x_2d @ w_k.T).view(B * S, H_KV, D)
    v = (x_2d @ w_v.T).view(B * S, H_KV, D)

    # ② fused QK-Norm + RoPE (1 Triton kernel launch)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (B * S, H_Q + H_KV)
    _qk_norm_rope_kernel[grid](
        q, k, q_out, k_out,
        q_norm_w, k_norm_w,
        cos, sin,
        S, H_Q,
        q.stride(0),   # stride_qt = H_Q * D
        k.stride(0),   # stride_kt = H_KV * D
        D=D, HALF_D=HALF_D, EPS=1e-6,
    )

    return (
        q_out.view(B, S, H_Q,  D),
        k_out.view(B, S, H_KV, D),
        v.view(B, S, H_KV, D),
    )
