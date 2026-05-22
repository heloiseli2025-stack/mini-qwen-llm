"""Fused QKV + QK-Norm + RoPE Kernel 测试（M2 验收）。

运行方式（云端 4090）：
    pytest tests/test_fused_qkv_rope.py -v -s
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")


def _make_weights(B, S, H_hidden, H_Q, H_KV, D, device, dtype=torch.bfloat16):
    """生成随机权重和 RoPE 表。"""
    x        = torch.randn(B, S, H_hidden, dtype=dtype, device=device)
    w_q      = torch.randn(H_Q * D,  H_hidden, dtype=dtype, device=device) * 0.02
    w_k      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    w_v      = torch.randn(H_KV * D, H_hidden, dtype=dtype, device=device) * 0.02
    q_norm_w = torch.ones(D, dtype=dtype, device=device)
    k_norm_w = torch.ones(D, dtype=dtype, device=device)

    # 真实 RoPE（从 RotaryEmbedding 获取）
    from mini_qwen.model.layers.rope import RotaryEmbedding
    rope = RotaryEmbedding(head_dim=D, max_seq_len=S + 1)
    cos, sin = rope.forward(seq_len=S)                 # [S, D] float32
    cos = cos.to(dtype=dtype, device=device)
    sin = sin.to(dtype=dtype, device=device)

    return x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin


def _reference_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    """PyTorch 参考实现，与 kernel 对齐：norm+rope 全程 fp32，最终转 bf16。

    Kernel 路径：GEMM(bf16) → load→fp32 → norm(fp32) → rope(fp32) → store→bf16
    若参照在 norm 后转 bf16 再做 rope，会多一次中间取整，导致 0.02~0.03 误差。
    """
    B, S, H = x.shape
    D    = q_norm_w.shape[0]
    H_Q  = w_q.shape[0] // D
    H_KV = w_k.shape[0] // D

    x_2d = x.reshape(B * S, H)
    q_raw = (x_2d @ w_q.T).view(B, S, H_Q,  D)   # bf16（与 kernel 一致）
    k_raw = (x_2d @ w_k.T).view(B, S, H_KV, D)
    v_raw = (x_2d @ w_v.T).view(B, S, H_KV, D)

    # RMSNorm：fp32，不提前转 bf16（与 kernel 保持一致）
    def _rms_norm_fp32(t, weight):
        tf = t.float()
        rms = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)
        return tf * rms * weight.float()   # 返回 fp32

    q_n = _rms_norm_fp32(q_raw, q_norm_w)   # fp32
    k_n = _rms_norm_fp32(k_raw, k_norm_w)

    # rotate_half（fp32）
    def _rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    # RoPE：fp32，最终转 bf16
    cos_f = cos.float().unsqueeze(0).unsqueeze(2)   # [1, S, 1, D]
    sin_f = sin.float().unsqueeze(0).unsqueeze(2)
    q_out = (q_n * cos_f + _rotate_half(q_n) * sin_f).to(torch.bfloat16)
    k_out = (k_n * cos_f + _rotate_half(k_n) * sin_f).to(torch.bfloat16)

    return q_out, k_out, v_raw


# ── 正确性测试 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,S,H_hidden,H_Q,H_KV,D", [
    (1,  32,  1024, 16, 8, 128),
    (2,  64,  1024, 16, 8, 128),
    (4, 128,  1024, 16, 8, 128),
])
def test_m2_correctness(B, S, H_hidden, H_Q, H_KV, D):
    """Q/K 输出与 PyTorch 参考实现 max abs error < 1e-2（bf16 精度限制）。"""
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
    """V 必须完全等于 GEMM 输出（无 norm/rope），误差为 0。"""
    _require_cuda()
    from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope

    B, S, H_hidden, H_Q, H_KV, D = 2, 64, 1024, 16, 8, 128
    device = "cuda"
    x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin = _make_weights(
        B, S, H_hidden, H_Q, H_KV, D, device
    )

    _, _, v_out = fused_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin)

    # 直接计算期望的 V
    x_2d = x.reshape(B * S, H_hidden)
    v_ref = (x_2d @ w_v.T).view(B, S, H_KV, D)

    v_err = (v_out.float() - v_ref.float()).abs().max().item()
    print(f"\n  V passthrough max_err={v_err:.6f}")
    assert v_err == 0.0, f"V 不等于 GEMM 输出，误差 {v_err}"


# ── 性能测试 ──────────────────────────────────────────────────────────────────

def test_m2_perf():
    """打印 fused vs unfused GPU 时间，并记录 kernel 数（使用 torch.profiler）。"""
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
        # RMSNorm Q
        q_fp = q.float()
        q = (q_fp * torch.rsqrt(q_fp.pow(2).mean(-1, keepdim=True) + eps)
             * q_norm_w.float()).to(torch.bfloat16)
        # RMSNorm K
        k_fp = k.float()
        k = (k_fp * torch.rsqrt(k_fp.pow(2).mean(-1, keepdim=True) + eps)
             * k_norm_w.float()).to(torch.bfloat16)
        # RoPE
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        q = q * cos_b + rotate_half(q) * sin_b
        k = k * cos_b + rotate_half(k) * sin_b
        return q, k, v

    def fused():
        return fused_qkv_rope(x, w_q, w_k, w_v, q_norm_w, k_norm_w, cos, sin)

    # ── warmup ──
    for _ in range(WARMUP):
        unfused(); fused()
    torch.cuda.synchronize()

    # ── CUDA event 计时 ──
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

    # ── torch.profiler kernel 计数 ──
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof_u:
        unfused()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof_f:
        fused()
    torch.cuda.synchronize()

    # PyTorch 2.8 profiler：用 events() 逐条计数 CUDA kernel
    def count_cuda_kernels(prof):
        return sum(
            1 for e in prof.events()
            if getattr(e, "device_type", None) is not None
            and str(getattr(e, "device_type", "")) != "DeviceType.CPU"
        )

    n_unfused = count_cuda_kernels(prof_u)
    n_fused   = count_cuda_kernels(prof_f)
    print(f"  CUDA kernel launches: unfused={n_unfused}  fused={n_fused}")

    # 打印 profiler 表供肉眼确认（按 cpu 耗时排序）
    print("\n  [unfused profiler]")
    print(prof_u.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print("\n  [fused profiler]")
    print(prof_f.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
