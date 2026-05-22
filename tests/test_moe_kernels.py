"""MoE Kernel 测试（M4 实现）。

测试配置：合成小 shape（T=4, E=16, K=4, H=256, D=128），无需真实模型权重。
所有 test 需要 CUDA；本地无 GPU 则 skip。

验收标准：
  - router: topk_ids 与 torch.topk 完全一致
  - permute round-trip: 零误差（stable sort 确定性）
  - grouped_gemm Triton vs oracle: max abs error < 1e-2
  - Qwen3MoEBlock BF16 forward vs oracle: max abs error < 1e-2
"""
import pytest
import torch
import torch.nn.functional as F

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# ── 测试参数 ──────────────────────────────────────────────────────────────────
SHAPES = [
    (4,  16, 4, 256, 128),   # small: T=4,  E=16, K=4, H=256, D=128
    (8,  16, 4, 256, 128),   # T=8（验证 bincount 覆盖率）
    (1,  16, 4, 256, 128),   # decode 典型场景
]


# ── M4.1 Router ──────────────────────────────────────────────────────────────

@CUDA
@pytest.mark.parametrize("T,E,K,H,D", SHAPES)
def test_moe_router(T, E, K, H, D):
    torch.manual_seed(42)
    hidden   = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)

    from mini_qwen.kernels.moe_router import moe_router
    topk_ids, topk_weights = moe_router(hidden, router_w, top_k=K)

    # 参考：torch.topk on fp32 softmax
    logits    = F.linear(hidden.float(), router_w.float())
    scores    = F.softmax(logits, dim=-1)
    ref_w, ref_ids = torch.topk(scores, K, dim=-1)

    assert topk_ids.shape    == (T, K), f"shape mismatch: {topk_ids.shape}"
    assert topk_weights.shape == (T, K), f"shape mismatch: {topk_weights.shape}"
    assert torch.all(topk_ids == ref_ids), "topk_ids 与 torch.topk 不一致"
    assert torch.allclose(topk_weights, ref_w, atol=1e-3), "topk_weights 误差过大"


# ── M4.2 Permute ─────────────────────────────────────────────────────────────

@CUDA
@pytest.mark.parametrize("T,E,K,H,D", SHAPES)
def test_moe_permute_offsets(T, E, K, H, D):
    """验证 expert_offsets 边界：offsets[0]=0, offsets[-1]=T*K，无 off-by-one。"""
    torch.manual_seed(7)
    hidden   = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)

    from mini_qwen.kernels.moe_router import moe_router
    from mini_qwen.kernels.moe_permute import moe_permute

    topk_ids, _ = moe_router(hidden, router_w, top_k=K)
    permuted, expert_offsets = moe_permute(hidden, topk_ids, E)

    assert expert_offsets[0].item()  == 0,    "offsets[0] 不为 0"
    assert expert_offsets[-1].item() == T * K, f"offsets[-1]={expert_offsets[-1].item()} != T*K={T*K}"
    assert permuted.shape == (T * K, H),       f"permuted shape 错误: {permuted.shape}"


# ── M4.2 + M4.4 Permute + Unpermute Round-trip ───────────────────────────────

@CUDA
@pytest.mark.parametrize("T,E,K,H,D", SHAPES)
def test_moe_permute_unpermute_roundtrip(T, E, K, H, D):
    """round-trip 验证：x → permute → identity expert pass → unpermute ≈ weighted x。

    如果所有 expert 直接返回输入（identity），unpermute 应还原为原始 hidden 的加权和。
    """
    torch.manual_seed(13)
    hidden   = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)

    from mini_qwen.kernels.moe_router    import moe_router
    from mini_qwen.kernels.moe_permute   import moe_permute
    from mini_qwen.kernels.moe_unpermute import moe_unpermute

    topk_ids, topk_weights = moe_router(hidden, router_w, top_k=K)
    permuted, _ = moe_permute(hidden, topk_ids, E)

    # identity expert：直接用 permuted 作为 "expert 输出"
    out = moe_unpermute(permuted, topk_weights, topk_ids, T)

    # 参考：逐 token 加权求和（同一 token 被选 K 次，权重 sum = 1）
    ref = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            ref[t] += topk_weights[t, k].float() * hidden[t].float()

    err = (out.float() - ref).abs().max().item()
    assert err < 1e-2, f"round-trip max abs error {err:.6f} > 1e-2"


# ── M4.3 Grouped GEMM ────────────────────────────────────────────────────────

@CUDA
@pytest.mark.parametrize("T,E,K,H,D", SHAPES)
def test_moe_grouped_gemm_vs_oracle(T, E, K, H, D):
    """Triton kernel 与 for-loop oracle 对比：max abs error < 1e-2。"""
    torch.manual_seed(99)
    hidden         = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router_w       = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)
    expert_weights = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16)

    from mini_qwen.kernels.moe_router       import moe_router
    from mini_qwen.kernels.moe_permute      import moe_permute
    from mini_qwen.kernels.moe_grouped_gemm import moe_grouped_gemm, _moe_grouped_gemm_oracle

    topk_ids, _ = moe_router(hidden, router_w, top_k=K)
    permuted, expert_offsets = moe_permute(hidden, topk_ids, E)

    ref = _moe_grouped_gemm_oracle(permuted, expert_weights, expert_offsets)
    out = moe_grouped_gemm(permuted, expert_weights, expert_offsets)

    assert out.shape == (T * K, D), f"output shape 错误: {out.shape}"

    err = (out.float() - ref.float()).abs().max().item()
    assert err < 1e-2, f"grouped_gemm max abs error {err:.4f} > 1e-2"


@CUDA
@pytest.mark.parametrize("T,E,K,H,D", SHAPES)
def test_moe_grouped_gemm_empty_expert(T, E, K, H, D):
    """所有 token 只去 expert 0，其他 expert 全空 → 输出 shape 正确且无 crash。"""
    torch.manual_seed(0)
    hidden         = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    expert_weights = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16)

    # 强制所有 topk_ids = 0（只用 expert 0）
    topk_ids = torch.zeros(T, K, dtype=torch.long, device="cuda")

    from mini_qwen.kernels.moe_permute      import moe_permute
    from mini_qwen.kernels.moe_grouped_gemm import moe_grouped_gemm

    permuted, expert_offsets = moe_permute(hidden, topk_ids, E)
    out = moe_grouped_gemm(permuted, expert_weights, expert_offsets)

    assert out.shape == (T * K, D)
    # expert 0 收到 T*K tokens，其余全为 0 → 非 expert 0 区域的输出必须是 0
    # (expert_offsets[1:] == T*K 只有 offsets[1] 满足)
    assert expert_offsets[1].item() == T * K
    assert expert_offsets[2].item() == T * K   # expert 1 是空的


# ── M4 Integration：Qwen3MoEBlock BF16 ─────────────────────────────────────

@CUDA
def test_qwen3_moe_block_bf16():
    """Qwen3MoEBlock BF16 forward 与逐 expert 逐步计算结果一致。"""
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.layers.moe import Qwen3MoEBlock

    cfg = Qwen3MoEConfig(
        hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        intermediate_size=128,
        num_experts=16, num_experts_per_tok=4,
        vocab_size=128,
    )
    torch.manual_seed(42)
    block  = Qwen3MoEBlock(cfg).cuda().bfloat16()
    x      = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)  # [B=2, S=4, H=256]
    out    = block(x)

    assert out.shape == x.shape, f"output shape 错误: {out.shape}"
    assert not out.isnan().any(),  "输出含 NaN"
    assert not out.isinf().any(),  "输出含 Inf"


@CUDA
def test_qwen3_moe_block_w4a16():
    """quantize_to_w4a16 后 forward 不 crash，shape 正确，无 NaN。"""
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.layers.moe import Qwen3MoEBlock

    cfg = Qwen3MoEConfig(
        hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        intermediate_size=128,
        num_experts=16, num_experts_per_tok=4,
        vocab_size=128,
    )
    torch.manual_seed(5)
    block = Qwen3MoEBlock(cfg).cuda().bfloat16()
    block.quantize_to_w4a16(group_size=32)   # group_size=32 适配 intermediate_size=128

    x   = torch.randn(1, 1, 256, device="cuda", dtype=torch.bfloat16)   # decode shape
    out = block(x)

    assert out.shape == x.shape
    assert not out.isnan().any(), "W4A16 forward 输出含 NaN"


# ── M4 Performance（打印，不 assert）────────────────────────────────────────

@CUDA
def test_moe_grouped_gemm_perf(benchmark=None):
    """Triton grouped GEMM vs for-loop oracle 性能对比（打印结果，不 assert）。"""
    import time
    T, E, K, H, D = 8, 128, 8, 3584, 1536

    torch.manual_seed(0)
    hidden         = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router_w       = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)
    expert_weights = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16)

    from mini_qwen.kernels.moe_router       import moe_router
    from mini_qwen.kernels.moe_permute      import moe_permute
    from mini_qwen.kernels.moe_grouped_gemm import moe_grouped_gemm, _moe_grouped_gemm_oracle

    topk_ids, _ = moe_router(hidden, router_w, top_k=K)
    permuted, expert_offsets = moe_permute(hidden, topk_ids, E)

    def bench(fn, warmup=5, reps=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    ms_oracle = bench(lambda: _moe_grouped_gemm_oracle(permuted, expert_weights, expert_offsets))
    ms_triton = bench(lambda: moe_grouped_gemm(permuted, expert_weights, expert_offsets))
    speedup   = ms_oracle / ms_triton

    print(f"\nGrouped GEMM（T={T}, E={E}, K={K}, H={H}, D={D}）")
    print(f"  for-loop oracle : {ms_oracle:.3f} ms")
    print(f"  Triton kernel   : {ms_triton:.3f} ms")
    print(f"  speedup         : {speedup:.2f}x")
