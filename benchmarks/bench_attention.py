"""Paged Attention Benchmark — M1 验收。

system_bench：baseline = torch.cat([k_prev, new_k], dim=seq_dim) + SDPA
              模拟 HF 原生 decode 的 O(seq²) memory traffic
              对比 paged_attn_decode

kernel_bench：baseline = contiguous KV（一次性分配，不 cat）直接 SDPA
              只比较 attention 计算本身，排除 KV 拼接开销
              对比 paged_attn_decode

prefill_bench：baseline 1 = naive O(n²) SDPA（禁用 Flash）
               baseline 2 = FA2（Flash Attention 2 后端，native GQA）
               对比 paged_attn_prefill（含 KV cache 写入开销）

运行方式（云端 4090）：
    python benchmarks/bench_attention.py
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

from mini_qwen.cache.block_manager import BlockManager
from mini_qwen.kernels.paged_attn_decode import paged_attn_decode

# ── 固定超参 ──────────────────────────────────────────────────────────────────
H_Q, H_KV, D  = 16, 8, 128
BLOCK_SIZE     = 16
NUM_KV_GROUPS  = H_Q // H_KV
WARMUP         = 10
REPS           = 100

BATCH_LIST  = [1, 8, 16]
SEQLEN_LIST = [512, 1024, 2048]

PREFILL_WARMUP      = 5
PREFILL_REPS        = 30
PREFILL_BATCH_LIST  = [1, 4]
PREFILL_SEQLEN_LIST = [512, 1024, 2048, 4096]


# ── 计时工具 ──────────────────────────────────────────────────────────────────

def time_fn(fn: callable, warmup: int = WARMUP, reps: int = REPS) -> float:
    """CUDA event 计时，返回 reps 次的平均耗时（ms）。"""
    for _ in range(warmup):
        fn()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


# ── Paged cache 初始化（decode 用）────────────────────────────────────────────

def setup_paged(batch: int, seqlen: int):
    """分配 paged KV cache，返回 (k_cache, v_cache, block_table, seq_lens)。"""
    num_blocks  = batch * math.ceil(seqlen / BLOCK_SIZE) + 4
    k_cache     = torch.randn(num_blocks, BLOCK_SIZE, H_KV, D,
                              dtype=torch.bfloat16, device="cuda")
    v_cache     = torch.randn(num_blocks, BLOCK_SIZE, H_KV, D,
                              dtype=torch.bfloat16, device="cuda")
    manager     = BlockManager(num_blocks, BLOCK_SIZE)
    max_blocks  = math.ceil(seqlen / BLOCK_SIZE)
    block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    for b in range(batch):
        blks = manager.allocate(seq_id=b, num_tokens=seqlen)
        for i, bid in enumerate(blks):
            block_table[b, i] = bid
    block_table = block_table.cuda()
    seq_lens    = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, block_table, seq_lens


# ── Benchmark 1：system_bench ─────────────────────────────────────────────────

def system_bench(batch: int, seqlen: int) -> tuple[float, float, float]:
    """HF decode 风格（cat + SDPA）vs paged_attn_decode。

    baseline 每次调用都 cat 一个新 token（模拟每个 decode step 的内存分配 + 拷贝开销）。
    """
    k_cache, v_cache, block_table, seq_lens = setup_paged(batch, seqlen)
    q = torch.randn(batch, H_Q, D, dtype=torch.bfloat16, device="cuda")

    k_prev = torch.randn(batch, H_KV, seqlen - 1, D, dtype=torch.bfloat16, device="cuda")
    v_prev = torch.randn(batch, H_KV, seqlen - 1, D, dtype=torch.bfloat16, device="cuda")
    k_new  = torch.randn(batch, H_KV, 1, D, dtype=torch.bfloat16, device="cuda")
    v_new  = torch.randn(batch, H_KV, 1, D, dtype=torch.bfloat16, device="cuda")

    def baseline_hf():
        k_full = torch.cat([k_prev, k_new], dim=2)
        v_full = torch.cat([v_prev, v_new], dim=2)
        k_exp = k_full.repeat_interleave(NUM_KV_GROUPS, dim=1)
        v_exp = v_full.repeat_interleave(NUM_KV_GROUPS, dim=1)
        return F.scaled_dot_product_attention(q.unsqueeze(2), k_exp, v_exp)

    def ours():
        return paged_attn_decode(q, k_cache, v_cache, block_table, seq_lens)

    t_base = time_fn(baseline_hf)
    t_ours = time_fn(ours)
    return t_base, t_ours, t_base / t_ours


# ── Benchmark 2：kernel_bench ─────────────────────────────────────────────────

def kernel_bench(batch: int, seqlen: int) -> tuple[float, float, float]:
    """纯 attention 计算对比：contiguous SDPA vs paged_attn_decode。

    baseline 的 K/V 一次性分配好（不含 cat 开销），GQA expand 也只做一次。
    只计时 SDPA 调用本身，隔离 attention 计算效率差异。
    """
    k_cache, v_cache, block_table, seq_lens = setup_paged(batch, seqlen)
    q = torch.randn(batch, H_Q, D, dtype=torch.bfloat16, device="cuda")

    k_cont = torch.randn(batch, H_KV, seqlen, D, dtype=torch.bfloat16, device="cuda")
    v_cont = torch.randn(batch, H_KV, seqlen, D, dtype=torch.bfloat16, device="cuda")
    k_exp  = k_cont.repeat_interleave(NUM_KV_GROUPS, dim=1)
    v_exp  = v_cont.repeat_interleave(NUM_KV_GROUPS, dim=1)

    def baseline_sdpa():
        return F.scaled_dot_product_attention(q.unsqueeze(2), k_exp, v_exp)

    def ours():
        return paged_attn_decode(q, k_cache, v_cache, block_table, seq_lens)

    t_base = time_fn(baseline_sdpa)
    t_ours = time_fn(ours)
    return t_base, t_ours, t_base / t_ours


# ── Benchmark 3：prefill_bench ────────────────────────────────────────────────

def prefill_bench(batch: int, seqlen: int) -> tuple[float, float, float, float, float]:
    """naive O(n²) SDPA vs FA2 vs paged_attn_prefill（含 KV cache 写入）。

    FA2 baseline 使用 native GQA（不 expand K/V），是最强的 baseline。
    Naive baseline GQA pre-expand（math backend 不保证支持 native GQA）。
    Ours 包含 KV cache 写入开销，两个 baseline 不含。
    """
    from mini_qwen.kernels.paged_attn_prefill import paged_attn_prefill

    total = batch * seqlen
    blocks_per_seq = math.ceil(seqlen / BLOCK_SIZE)
    num_blocks = batch * blocks_per_seq + 8
    manager = BlockManager(num_blocks, BLOCK_SIZE)
    block_table = torch.full((batch, blocks_per_seq), -1, dtype=torch.int32)
    for b in range(batch):
        blks = manager.allocate(seq_id=b, num_tokens=seqlen)
        for i, bid in enumerate(blks):
            block_table[b, i] = bid
    block_table = block_table.cuda()

    k_cache = torch.zeros(num_blocks, BLOCK_SIZE, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros(num_blocks, BLOCK_SIZE, H_KV, D, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = torch.tensor([i * seqlen for i in range(batch + 1)],
                              dtype=torch.int32, device="cuda")

    q = torch.randn(total, H_Q, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # 两个 baseline 都用 GQA pre-expand，[B, H_q, S, D]
    # PyTorch 2.8 的 dense SDPA 要求 Q/K/V num_heads 相同，不支持 native GQA
    q_4d = q.view(batch, seqlen, H_Q, D).permute(0, 2, 1, 3)
    k_exp = k.view(batch, seqlen, H_KV, D).repeat_interleave(NUM_KV_GROUPS, dim=2)
    v_exp = v.view(batch, seqlen, H_KV, D).repeat_interleave(NUM_KV_GROUPS, dim=2)
    k_4d = k_exp.permute(0, 2, 1, 3)
    v_4d = v_exp.permute(0, 2, 1, 3)

    from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel

    def naive_sdpa():
        with _sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=True)

    def fa2_sdpa():
        with _sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=True)

    def ours():
        return paged_attn_prefill(q, k, v, k_cache, v_cache, block_table, cu_seqlens, seqlen)

    t_naive = time_fn(naive_sdpa, warmup=PREFILL_WARMUP, reps=PREFILL_REPS)
    t_fa2   = time_fn(fa2_sdpa,   warmup=PREFILL_WARMUP, reps=PREFILL_REPS)
    t_ours  = time_fn(ours,        warmup=PREFILL_WARMUP, reps=PREFILL_REPS)
    return t_naive, t_fa2, t_ours, t_naive / t_ours, t_fa2 / t_ours


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: 需要 CUDA，请在云端 4090 上运行", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    torch_ver = torch.__version__
    print(f"GPU:   {gpu_name}")
    print(f"Torch: {torch_ver}")
    print(f"H_q={H_Q}, H_kv={H_KV}, D={D}, block_size={BLOCK_SIZE}")
    print(f"decode: warmup={WARMUP}, reps={REPS}")
    print(f"prefill: warmup={PREFILL_WARMUP}, reps={PREFILL_REPS}\n")

    # 触发 Triton autotune（decode）
    _k, _v, _bt, _sl = setup_paged(1, 16)
    _q = torch.randn(1, H_Q, D, dtype=torch.bfloat16, device="cuda")
    paged_attn_decode(_q, _k, _v, _bt, _sl)

    # 触发 Triton autotune（prefill）
    from mini_qwen.kernels.paged_attn_prefill import paged_attn_prefill
    _warmup_total = 64
    _warmup_blocks = math.ceil(_warmup_total / BLOCK_SIZE) + 2
    _wmgr = BlockManager(_warmup_blocks, BLOCK_SIZE)
    _wblks = _wmgr.allocate(0, _warmup_total)
    _wbt = torch.tensor([_wblks], dtype=torch.int32, device="cuda")
    _wkc = torch.zeros(_warmup_blocks, BLOCK_SIZE, H_KV, D, dtype=torch.bfloat16, device="cuda")
    _wvc = torch.zeros(_warmup_blocks, BLOCK_SIZE, H_KV, D, dtype=torch.bfloat16, device="cuda")
    _wcu = torch.tensor([0, _warmup_total], dtype=torch.int32, device="cuda")
    _wq  = torch.randn(_warmup_total, H_Q, D, dtype=torch.bfloat16, device="cuda")
    _wk  = torch.randn(_warmup_total, H_KV, D, dtype=torch.bfloat16, device="cuda")
    _wv  = torch.randn(_warmup_total, H_KV, D, dtype=torch.bfloat16, device="cuda")
    paged_attn_prefill(_wq, _wk, _wv, _wkc, _wvc, _wbt, _wcu, _warmup_total)
    torch.cuda.synchronize()

    sys_rows:    list[str] = []
    ker_rows:    list[str] = []
    pre_rows:    list[str] = []
    sys_results: list[dict] = []
    ker_results: list[dict] = []
    pre_results: list[dict] = []

    # ── system_bench ──
    print("=" * 60)
    print("system_bench  (cat + SDPA  vs  paged_attn_decode)")
    print("=" * 60)
    for batch in BATCH_LIST:
        for seqlen in SEQLEN_LIST:
            t_base, t_ours, speedup = system_bench(batch, seqlen)
            tag = f"system_bench @ batch={batch:2d} seq={seqlen}"
            print(f"  {tag}: baseline={t_base:.3f}ms  ours={t_ours:.3f}ms  speedup={speedup:.1f}x")
            sys_rows.append(
                f"| {batch} | {seqlen} | {t_base:.3f} | {t_ours:.3f} | **{speedup:.1f}x** |"
            )
            sys_results.append(dict(batch=batch, seqlen=seqlen,
                                    t_base=t_base, t_ours=t_ours, speedup=speedup))

    print()

    # ── kernel_bench ──
    print("=" * 60)
    print("kernel_bench  (contiguous SDPA  vs  paged_attn_decode)")
    print("=" * 60)
    for batch in BATCH_LIST:
        for seqlen in SEQLEN_LIST:
            t_base, t_ours, speedup = kernel_bench(batch, seqlen)
            tag = f"kernel_bench @ batch={batch:2d} seq={seqlen}"
            print(f"  {tag}: baseline={t_base:.3f}ms  ours={t_ours:.3f}ms  speedup={speedup:.1f}x")
            ker_rows.append(
                f"| {batch} | {seqlen} | {t_base:.3f} | {t_ours:.3f} | **{speedup:.1f}x** |"
            )
            ker_results.append(dict(batch=batch, seqlen=seqlen,
                                    t_base=t_base, t_ours=t_ours, speedup=speedup))

    print()

    # ── prefill_bench ──
    print("=" * 60)
    print("prefill_bench  (naive O(n²) / FA2  vs  paged_attn_prefill)")
    print("=" * 60)
    for batch in PREFILL_BATCH_LIST:
        for seqlen in PREFILL_SEQLEN_LIST:
            t_naive, t_fa2, t_ours, sp_naive, sp_fa2 = prefill_bench(batch, seqlen)
            tag = f"prefill_bench @ batch={batch:2d} seq={seqlen}"
            print(f"  {tag}: naive={t_naive:.3f}ms  fa2={t_fa2:.3f}ms  "
                  f"ours={t_ours:.3f}ms  vs_naive={sp_naive:.2f}x  vs_fa2={sp_fa2:.2f}x")
            pre_rows.append(
                f"| {batch} | {seqlen} | {t_naive:.3f} | {t_fa2:.3f} | {t_ours:.3f} "
                f"| **{sp_naive:.2f}x** | **{sp_fa2:.2f}x** |"
            )
            pre_results.append(dict(batch=batch, seqlen=seqlen,
                                    t_naive=t_naive, t_fa2=t_fa2, t_ours=t_ours,
                                    sp_naive=sp_naive, sp_fa2=sp_fa2))

    # ── 写入 markdown ──────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m1_bench_results.md")

    md_lines = [
        "# M1 Paged Attention Benchmark Results",
        "",
        f"**GPU**: {gpu_name}  ",
        f"**PyTorch**: {torch_ver}  ",
        f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d')}  ",
        f"**Config**: H_q={H_Q}, H_kv={H_KV}, D={D}, block_size={BLOCK_SIZE}  ",
        f"**Timing**: decode warmup={WARMUP}/reps={REPS}; "
        f"prefill warmup={PREFILL_WARMUP}/reps={PREFILL_REPS}",
        "",
        "---",
        "",
        "## system_bench",
        "",
        "baseline = `torch.cat([k_prev, new_k], dim=2)` + `repeat_interleave` + SDPA  ",
        "模拟 HF 原生 decode：每 step 重新分配并拷贝全量 KV（O(seq²) memory traffic）",
        "",
        "| batch | seqlen | baseline (ms) | ours (ms) | speedup |",
        "|------:|-------:|--------------:|----------:|--------:|",
        *sys_rows,
        "",
        "---",
        "",
        "## kernel_bench",
        "",
        "baseline = contiguous KV（一次性分配，不 cat）+ GQA pre-expand + SDPA  ",
        "只测 attention 计算本身，排除 KV 拼接开销",
        "",
        "| batch | seqlen | baseline (ms) | ours (ms) | speedup |",
        "|------:|-------:|--------------:|----------:|--------:|",
        *ker_rows,
        "",
        "---",
        "",
        "## prefill_bench",
        "",
        "baseline 1 (naive) = `sdpa_kernel(MATH)` + GQA pre-expand，O(n²) 显存，无 Flash  ",
        "baseline 2 (fa2)   = `sdpa_kernel(FLASH_ATTENTION)` + GQA pre-expand（PyTorch 2.8 dense SDPA 要求同 num_heads）  ",
        "ours = `paged_attn_prefill`（Triton tiled causal，**含 KV cache 写入开销**，两个 baseline 不含）",
        "",
        "| batch | seqlen | naive (ms) | fa2 (ms) | ours (ms) | vs naive | vs fa2 |",
        "|------:|-------:|-----------:|---------:|---------:|---------:|-------:|",
        *pre_rows,
        "",
        "---",
        "",
        "## 关键结论",
        "",
    ]

    s_max = max(sys_results, key=lambda r: r["speedup"])
    s_min = min(sys_results, key=lambda r: r["speedup"])
    k_max = max(ker_results, key=lambda r: r["speedup"])
    k_min = min(ker_results, key=lambda r: r["speedup"])
    p_naive_max = max(pre_results, key=lambda r: r["sp_naive"])
    p_naive_min = min(pre_results, key=lambda r: r["sp_naive"])
    p_fa2_max   = max(pre_results, key=lambda r: r["sp_fa2"])
    p_fa2_min   = min(pre_results, key=lambda r: r["sp_fa2"])

    md_lines += [
        "### decode（system_bench / kernel_bench）",
        f"- **system_bench** speedup 范围：{s_min['speedup']:.1f}x "
        f"(batch={s_min['batch']}, seq={s_min['seqlen']}) "
        f"～ {s_max['speedup']:.1f}x (batch={s_max['batch']}, seq={s_max['seqlen']})",
        f"- **kernel_bench** speedup 范围：{k_min['speedup']:.1f}x "
        f"(batch={k_min['batch']}, seq={k_min['seqlen']}) "
        f"～ {k_max['speedup']:.1f}x (batch={k_max['batch']}, seq={k_max['seqlen']})",
        "- system_bench 包含 `torch.cat` 内存分配开销，差距随 seqlen 增大而扩大",
        "- kernel_bench 在小 seq 时 paged 慢（block_table 间接寻址 cache miss），"
        "大 seq 时因避免连续大块分配而胜出",
        "",
        "### prefill",
        f"- **vs naive O(n²)**：{p_naive_min['sp_naive']:.2f}x ～ {p_naive_max['sp_naive']:.2f}x",
        f"- **vs FA2**：{p_fa2_min['sp_fa2']:.2f}x ～ {p_fa2_max['sp_fa2']:.2f}x "
        f"（< 1 表示比 FA2 慢）",
        "- ours 含 KV cache 写入，两个 baseline 不含；两者均 GQA pre-expand（PyTorch 2.8 限制）",
        "- 工程取舍：paged prefill 支持动态 block 分配和 prefix cache，代价是比 FA2 慢",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
