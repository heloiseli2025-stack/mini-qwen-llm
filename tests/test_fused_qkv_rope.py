"""Fused QKV + QK-Norm + RoPE Kernel tests (M2 acceptance).

Run (cloud GPU):
    pytest tests/test_fused_qkv_rope.py -v -s
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _make_weights(B, S, H_hidden, H_Q, H_KV, D, device, dtype=torch.bfloat16):
    """Random weights and RoPE tables for prefill (S tokens, uniform length)."""
    x        = torch.randn(B, S, H_hidden, dtype=dtype, device=device)
    w_q      = torch.randn(H_Q * D,  H_hidden, dtype=dtype, device=device) * 0.02
    w_k      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    w_v      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    q_norm_w = torch.ones(D, dtype=dtype, device=device)
    k_norm_w = torch.ones(D, dtype=dtype, device=device)

    from mini_qwen.model.layers.rope import RotaryEmbedding
    rope = RotaryEmbedding(head_dim=D, max_seq_len=S + 1)
    cos, sin = rope.forward(seq_len=S)   # [S, D] float32
    cos = cos.to(dtype=dtype, device=device)
    sin = sin.to(dtype=dtype, device=device)

    return x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin


def _reference_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    """PyTorch oracle matching kernel numerics: norm+rope in fp32, output in bf16.

    Kernel path: GEMM(bf16) → load→fp32 → norm(fp32) → rope(fp32) → store→bf16.
    Keeping norm and rope in fp32 avoids an extra rounding that would push
    max-abs error above the 1e-2 bf16 threshold.
    """
    B, S, H = x.shape
    D    = q_norm_w.shape[0]
    H_Q  = w_q.shape[0] // D
    H_KV = w_k.shape[0] // D

    x_2d = x.reshape(B * S, H)
    q_raw = (x_2d @ w_q.T).view(B, S, H_Q,  D)
    k_raw = (x_2d @ w_k.T).view(B, S, H_KV, D)
    v_raw = (x_2d @ w_v.T).view(B, S, H_KV, D)

    def _rms_norm_fp32(t, weight):
        tf = t.float()
        rms = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)
        return tf * rms * weight.float()

    def _rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    q_n = _rms_norm_fp32(q_raw, q_norm_w)
    k_n = _rms_norm_fp32(k_raw, k_norm_w)

    cos_f = cos.float().unsqueeze(0).unsqueeze(2)   # [1, S, 1, D]
    sin_f = sin.float().unsqueeze(0).unsqueeze(2)
    q_out = (q_n * cos_f + _rotate_half(q_n) * sin_f).to(torch.bfloat16)
    k_out = (k_n * cos_f + _rotate_half(k_n) * sin_f).to(torch.bfloat16)

    return q_out, k_out, v_raw


def _reference_qkv_rope_decode(x, w_q, w_k, w_v, q_norm_w, k_norm_w,
                                cos_per_seq, sin_per_seq, eps=1e-6):
    """Oracle for decode mode: x [B,1,H], cos/sin [B,D] pre-indexed per sequence."""
    B, _, H = x.shape
    D    = q_norm_w.shape[0]
    H_Q  = w_q.shape[0] // D
    H_KV = w_k.shape[0] // D

    x_2d = x.reshape(B, H)
    q_raw = (x_2d @ w_q.T).view(B, 1, H_Q,  D)
    k_raw = (x_2d @ w_k.T).view(B, 1, H_KV, D)
    v_raw = (x_2d @ w_v.T).view(B, 1, H_KV, D)

    def _rms_norm_fp32(t, weight):
        tf = t.float()
        rms = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)
        return tf * rms * weight.float()

    def _rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    q_n = _rms_norm_fp32(q_raw, q_norm_w)
    k_n = _rms_norm_fp32(k_raw, k_norm_w)

    # cos_per_seq [B, D] -> [B, 1, 1, D]
    cos_f = cos_per_seq.float().unsqueeze(1).unsqueeze(1)
    sin_f = sin_per_seq.float().unsqueeze(1).unsqueeze(1)
    q_out = (q_n * cos_f + _rotate_half(q_n) * sin_f).to(torch.bfloat16)
    k_out = (k_n * cos_f + _rotate_half(k_n) * sin_f).to(torch.bfloat16)

    return q_out, k_out, v_raw


# ── correctness tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,S,H_hidden,H_Q,H_KV,D", [
    (1,  32,  1024, 16, 8, 128),
    (2,  64,  1024, 16, 8, 128),
    (4, 128,  1024, 16, 8, 128),
])
def test_m2_correctness(B, S, H_hidden, H_Q, H_KV, D):
    """Q/K output max abs error < 1e-2 against PyTorch oracle (bf16 precision limit)."""
    _require_cuda()
    from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope

    device = "cuda"
    x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin = _make_weights(
        B, S, H_hidden, H_Q, H_KV, D, device
    )

    q_ref, k_ref, v_ref = _reference_qkv_rope(
        x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin
    )
    q_out, k_out, v_out = fused_qkv_rope(
        x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin
    )

    assert q_out.shape == (B, S, H_Q,  D), f"Q shape: {q_out.shape}"
    assert k_out.shape == (B, S, H_KV, D), f"K shape: {k_out.shape}"
    assert v_out.shape == (B, S, H_KV, D), f"V shape: {v_out.shape}"

    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()

    print(f"\n  B={B} S={S}: Q max_err={q_err:.4f}  K max_err={k_err:.4f}")

    assert q_err < 1e-2, f"Q max abs error {q_err:.4f} > 1e-2"
    assert k_err < 1e-2, f"K max abs error {k_err:.4f} > 1e-2"


def test_m2_v_passthrough():
    """V must equal GEMM output exactly (no norm/rope applied), error == 0."""
    _require_cuda()
    from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope

    B, S, H_hidden, H_Q, H_KV, D = 2, 64, 1024, 16, 8, 128
    device = "cuda"
    x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin = _make_weights(
        B, S, H_hidden, H_Q, H_KV, D, device
    )

    _, _, v_out = fused_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin)

    x_2d = x.reshape(B * S, H_hidden)
    v_ref = (x_2d @ w_v.T).view(B, S, H_KV, D)

    v_err = (v_out.float() - v_ref.float()).abs().max().item()
    print(f"\n  V passthrough max_err={v_err:.6f}")
    assert v_err == 0.0, f"V differs from GEMM output, err={v_err}"


@pytest.mark.parametrize("B,H_hidden,H_Q,H_KV,D", [
    (1, 1024, 16, 8, 128),
    (4, 1024, 16, 8, 128),
    (8, 1024, 16, 8, 128),
])
def test_m2_decode_mode(B, H_hidden, H_Q, H_KV, D):
    """Decode mode (positions != None): each token at a different RoPE position.

    Kernel receives cos/sin [B, D] pre-indexed per sequence and
    positions=arange(B), so token b uses row b of cos/sin (= cos_cached[pos_b]).
    """
    _require_cuda()
    from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope
    from mini_qwen.model.layers.rope import RotaryEmbedding

    device = "cuda"
    dtype  = torch.bfloat16

    x        = torch.randn(B, 1, H_hidden, dtype=dtype, device=device)
    w_q      = torch.randn(H_Q * D,  H_hidden, dtype=dtype, device=device) * 0.02
    w_k      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    w_v      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    q_norm_w = torch.ones(D, dtype=dtype, device=device)
    k_norm_w = torch.ones(D, dtype=dtype, device=device)

    # Different RoPE positions for each sequence (simulating mixed decode lengths)
    max_pos = 2048
    pos_vals = torch.randint(0, max_pos, (B,), device=device)
    rope = RotaryEmbedding(head_dim=D, max_seq_len=max_pos + 1)
    cos_full, sin_full = rope.forward(seq_len=max_pos)  # [max_pos, D]
    cos_full = cos_full.to(dtype=dtype, device=device)
    sin_full = sin_full.to(dtype=dtype, device=device)

    cos_per_seq = cos_full[pos_vals]   # [B, D]
    sin_per_seq = sin_full[pos_vals]   # [B, D]

    positions = torch.arange(B, device=device, dtype=torch.int32)

    q_ref, k_ref, v_ref = _reference_qkv_rope_decode(
        x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos_per_seq, sin_per_seq
    )
    q_out, k_out, v_out = fused_qkv_rope(
        x, w_q, w_k, w_v, q_norm_w, k_norm_w,
        cos_per_seq, sin_per_seq,
        positions=positions,
    )

    assert q_out.shape == (B, 1, H_Q,  D)
    assert k_out.shape == (B, 1, H_KV, D)

    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()
    print(f"\n  decode B={B}: Q max_err={q_err:.4f}  K max_err={k_err:.4f}")

    assert q_err < 1e-2, f"Decode Q max abs error {q_err:.4f} > 1e-2"
    assert k_err < 1e-2, f"Decode K max abs error {k_err:.4f} > 1e-2"


# ── performance tests ──────────────────────────────────────────────────────────

def test_m2_perf():
    """Print fused vs unfused GPU time and CUDA kernel launch count."""
    _require_cuda()
    from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope
    from mini_qwen.model.layers.rope import rotate_half

    B, S, H_hidden, H_Q, H_KV, D = 4, 512, 1024, 16, 8, 128
    WARMUP, REPS = 5, 30
    device = "cuda"

    x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin = _make_weights(
        B, S, H_hidden, H_Q, H_KV, D, device
    )

    eps = 1e-6

    def unfused():
        x_2d = x.reshape(B * S, H_hidden)
        q = (x_2d @ w_q.T).view(B, S, H_Q,  D)
        k = (x_2d @ w_k.T).view(B, S, H_KV, D)
        v = (x_2d @ w_v.T).view(B, S, H_KV, D)
        q_fp = q.float()
        q = (q_fp * torch.rsqrt(q_fp.pow(2).mean(-1, keepdim=True) + eps)
             * q_norm_w.float()).to(torch.bfloat16)
        k_fp = k.float()
        k = (k_fp * torch.rsqrt(k_fp.pow(2).mean(-1, keepdim=True) + eps)
             * k_norm_w.float()).to(torch.bfloat16)
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        q = q * cos_b + rotate_half(q) * sin_b
        k = k * cos_b + rotate_half(k) * sin_b
        return q, k, v

    def fused():
        return fused_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin)

    for _ in range(WARMUP):
        unfused(); fused()
    torch.cuda.synchronize()

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    e0.record()
    for _ in range(REPS):
        unfused()
    e1.record()
    torch.cuda.synchronize()
    t_unfused = e0.elapsed_time(e1) / REPS

    e0.record()
    for _ in range(REPS):
        fused()
    e1.record()
    torch.cuda.synchronize()
    t_fused = e0.elapsed_time(e1) / REPS

    print(f"\n  unfused={t_unfused:.3f}ms  fused={t_fused:.3f}ms  "
          f"speedup={t_unfused / t_fused:.2f}x")

    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof_u:
        unfused()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof_f:
        fused()
    torch.cuda.synchronize()

    def count_cuda_kernels(prof):
        return sum(
            1 for e in prof.events()
            if getattr(e, "device_type", None) is not None
            and str(getattr(e, "device_type", "")) != "DeviceType.CPU"
        )

    n_unfused = count_cuda_kernels(prof_u)
    n_fused   = count_cuda_kernels(prof_f)
    print(f"  CUDA kernel launches: unfused={n_unfused}  fused={n_fused}")

    print("\n  [unfused profiler]")
    print(prof_u.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print("\n  [fused profiler]")
    print(prof_f.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
