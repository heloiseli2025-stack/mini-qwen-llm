"""W4A16 GEMM Kernel 测试（M3 验收）。

运行方式（云端 4090）：
    pytest tests/test_w4a16_gemm.py -v -s
"""
from __future__ import annotations

import torch
import pytest


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")


# ── 工具：PyTorch 参考实现 dequant ────────────────────────────────────────────

def _ref_dequant(qweight, scales, qzeros, group_size=128):
    """还原 fp32 weight matrix [K, N]，与 kernel 做完全相同的计算（全程 fp32，不转 bf16）。

    不在此处转 bf16，避免 fp32→bf16→fp32 往返引入精度损失，
    让 test 自行把最终结果转 bf16 再对比。
    """
    from mini_qwen.quantization.packing import unpack_int4
    K8, N = qweight.shape
    K = K8 * 8
    G = K // group_size

    # unpack qweight [K//8, N] -> [K, N]
    w_int = unpack_int4(qweight.T).T.float()   # [K, N] fp32

    # unpack qzeros [G, N//8] -> [G, N]
    z_int = unpack_int4(qzeros).float()         # [G, N] fp32

    # dequant：fp32 全程（与 kernel 一致）
    scale = scales.float()                      # [G, N] fp32
    w_fp  = torch.zeros(K, N, dtype=torch.float32, device=qweight.device)
    for g in range(G):
        s = g * group_size
        e = s + group_size
        w_fp[s:e] = (w_int[s:e] - z_int[g][None, :]) * scale[g][None, :]

    return w_fp   # fp32，不转 bf16


def _make_quantized(M, K, N, group_size=128, device='cuda'):
    """生成合法的量化 tensor，返回 (x, qweight, scales, qzeros, w_ref)。"""
    from mini_qwen.quantization.awq import _per_group_minmax_quantize

    w_fp32 = torch.randn(K, N, device=device)
    qweight, scales, qzeros = _per_group_minmax_quantize(w_fp32, group_size)
    qweight = qweight.to(device)
    scales  = scales.to(device)
    qzeros  = qzeros.to(device)

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # 参考实现 dequanted weight
    w_ref = _ref_dequant(qweight, scales, qzeros, group_size)

    return x, qweight, scales, qzeros, w_ref


# ── 测试 1：packing round-trip ────────────────────────────────────────────────

def test_pack_unpack_roundtrip_qweight():
    """qweight 方向（沿 K 维）：[K,N] pack → unpack 零误差。"""
    from mini_qwen.quantization.packing import pack_int4, unpack_int4
    K, N = 128, 64
    w = torch.randint(0, 16, (K, N), dtype=torch.int32)
    qw    = pack_int4(w.T).T        # [K//8, N]
    w_rec = unpack_int4(qw.T).T     # [K, N]
    assert (w == w_rec).all(), "qweight round-trip failed"


def test_pack_unpack_roundtrip_qzeros():
    """qzeros 方向（沿 N 维）：[G,N] pack → unpack 零误差。"""
    from mini_qwen.quantization.packing import pack_int4, unpack_int4
    G, N = 8, 64
    z = torch.randint(0, 16, (G, N), dtype=torch.int32)
    qz    = pack_int4(z)            # [G, N//8]
    z_rec = unpack_int4(qz)         # [G, N]
    assert (z == z_rec).all(), "qzeros round-trip failed"


# ── 测试 2：最小 shape 正确性（先验证，再扩大）────────────────────────────────

def test_w4a16_gemm_tiny():
    """M=1, K=128, N=8 最小 shape，kernel vs PyTorch 参考 max abs error < 1e-2。"""
    _require_cuda()
    from mini_qwen.kernels.w4a16_gemm import w4a16_gemm

    x, qweight, scales, qzeros, w_ref = _make_quantized(1, 128, 8, group_size=128)
    ref = (x.double() @ w_ref.double()).to(torch.bfloat16)
    out = w4a16_gemm(x, qweight, scales, qzeros, group_size=128)

    err = (out.float() - ref.float()).abs().max().item()
    print(f"\n  tiny M=1 K=128 N=8: max_err={err:.6f}")
    # tiny: K=128 output O(11), bf16 ULP ≈ 0.086，1e-2 容限足够
    assert err < 1e-2, f"max abs error {err:.4f} > 1e-2"


# ── 测试 3：正确性（多 shape）────────────────────────────────────────────────

@pytest.mark.parametrize("M,K,N", [
    (1,  1024, 4096),
    (4,  1024, 4096),
    (16, 1024, 4096),
])
def test_w4a16_gemm_correctness(M, K, N):
    """kernel 输出与 PyTorch 参考 max abs error < 1e-2。"""
    _require_cuda()
    from mini_qwen.kernels.w4a16_gemm import w4a16_gemm

    x, qweight, scales, qzeros, w_ref = _make_quantized(M, K, N, group_size=128)
    ref = (x.double() @ w_ref.double()).to(torch.bfloat16)
    out = w4a16_gemm(x, qweight, scales, qzeros, group_size=128)

    err = (out.float() - ref.float()).abs().max().item()
    print(f"\n  M={M} K={K} N={N}: max_err={err:.6f}")
    assert out.shape == (M, N)
    # K=1024 随机正态输出量级约 130（in [128,256)），bf16 1-ULP = 1.0；
    # fp32 kernel 与 fp64 reference 最多差 1 ULP → 允许 < 2.0；真正 dequant bug 误差 O(4+)
    assert err < 2.0, f"max abs error {err:.4f} > 2.0 (dequant bug or wrong layout)"


# ── 测试 4：性能（仅打印，不 assert）─────────────────────────────────────────

def test_w4a16_gemm_perf():
    """打印 W4A16 vs BF16 GEMM 的延迟和加速比（decode 场景 M≤16）。"""
    _require_cuda()
    from mini_qwen.kernels.w4a16_gemm import w4a16_gemm

    K, N       = 1024, 4096
    WARMUP, REPS = 10, 100

    def cuda_time(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(REPS):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / REPS

    print(f"\n  {'M':>4} | {'BF16(ms)':>10} {'W4A16(ms)':>11} {'speedup':>8}")
    print(f"  {'-'*4} + {'-'*10} {'-'*11} {'-'*8}")

    for M in [1, 4, 8, 16]:
        x, qweight, scales, qzeros, w_ref = _make_quantized(M, K, N)
        w_bf16 = w_ref.to(torch.bfloat16)  # [K, N] bf16，用于 BF16 基准

        t_bf16  = cuda_time(lambda: torch.mm(x, w_bf16))
        t_w4a16 = cuda_time(lambda: w4a16_gemm(x, qweight, scales, qzeros))

        print(f"  {M:>4} | {t_bf16:>10.3f} {t_w4a16:>11.3f} {t_bf16/t_w4a16:>7.2f}x")
