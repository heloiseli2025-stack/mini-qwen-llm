"""Throughput comparison: mini-qwen-llm vs vLLM on the same model.

Requires:
    pip install vllm

Usage:
    python benchmarks/bench_vs_vllm.py \
        --model-path /root/autodl-tmp/Qwen3-30B-A3B-GPTQ-Int4 \
        --batch 1 8 16 \
        --prompt-len 512 \
        --gen-len 64

Both engines use the same prompt, batch size, and generation length.
Timing excludes model loading; uses CUDA events for wall-clock measurement.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


# ── mini-qwen-llm benchmark ───────────────────────────────────────────────────

def _bench_mini_qwen(model_path, batch_size, prompt_len, gen_len,
                     warmup, measure, device):
    from mini_qwen.cache.block_manager import BlockManager
    from mini_qwen.cache.kv_cache import KVCache, KVCacheConfig
    from mini_qwen.engine.runner import ModelRunner, generate_batch
    from mini_qwen.engine.scheduler import Scheduler
    from mini_qwen.model.loader import load_moe_from_gptq

    model = load_moe_from_gptq(model_path, device=device)
    model.eval()
    cfg = model.config

    num_blocks = 2048
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks, block_size=16,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
    )
    kv_caches = [KVCache(kv_cfg) for _ in range(cfg.num_hidden_layers)]

    base   = list(range(512))
    prompt = (base * ((prompt_len // 512) + 1))[:prompt_len]
    prompts = [prompt[:] for _ in range(batch_size)]

    def _reset():
        for kv in kv_caches:
            kv.k_cache.zero_()
            kv.v_cache.zero_()

    def _one_iter():
        _reset()
        bm     = BlockManager(num_blocks=num_blocks, block_size=16)
        sched  = Scheduler(bm, max_seqs_in_flight=batch_size + 4)
        runner = ModelRunner(model, kv_caches, bm)
        torch.cuda.synchronize()
        t0   = time.perf_counter()
        outs = generate_batch(runner, sched, prompts, max_new_tokens=gen_len)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return sum(len(v) for v in outs.values()) / elapsed

    for _ in range(warmup):
        _one_iter()
    tps_list = [_one_iter() for _ in range(measure)]

    del model, kv_caches
    torch.cuda.empty_cache()

    return statistics.median(tps_list)


# ── vLLM benchmark ────────────────────────────────────────────────────────────

def _bench_vllm(model_path, batch_size, prompt_len, gen_len, warmup, measure):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not installed. Run: pip install vllm")
        return None

    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=prompt_len + gen_len + 64,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(max_tokens=gen_len, temperature=0.0)

    prompt_text = " ".join(["hello"] * prompt_len)
    prompts_text = [prompt_text] * batch_size

    def _one_iter():
        torch.cuda.synchronize()
        t0  = time.perf_counter()
        out = llm.generate(prompts_text, sampling)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        return toks / elapsed

    for _ in range(warmup):
        _one_iter()
    tps_list = [_one_iter() for _ in range(measure)]

    del llm
    torch.cuda.empty_cache()

    return statistics.median(tps_list)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch",      type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len",    type=int, default=64)
    parser.add_argument("--warmup",     type=int, default=3)
    parser.add_argument("--measure",    type=int, default=10)
    parser.add_argument("--skip-vllm",  action="store_true",
                        help="Skip vLLM benchmark (only run mini-qwen-llm)")
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"\nModel:      {args.model_path}")
    print(f"prompt_len: {args.prompt_len}  gen_len: {args.gen_len}")
    print(f"warmup: {args.warmup}  measure: {args.measure}\n")

    header = f"{'batch':>6}  {'mini-qwen (tok/s)':>20}  {'vLLM (tok/s)':>14}  {'speedup':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for bs in args.batch:
        print(f"  batch={bs}: running mini-qwen-llm ...", flush=True)
        mq_tps = _bench_mini_qwen(
            args.model_path, bs, args.prompt_len, args.gen_len,
            args.warmup, args.measure, device,
        )
        print(f"    mini-qwen batch={bs}: {mq_tps:.1f} tok/s", flush=True)

        vllm_tps = None
        if not args.skip_vllm:
            print(f"  batch={bs}: running vLLM ...", flush=True)
            vllm_tps = _bench_vllm(
                args.model_path, bs, args.prompt_len, args.gen_len,
                args.warmup, args.measure,
            )
        results.append((bs, mq_tps, vllm_tps))

    print()
    print(header)
    print("-" * len(header))
    for bs, mq_tps, vllm_tps in results:
        if vllm_tps is not None:
            ratio = mq_tps / vllm_tps
            print(f"{bs:>6}  {mq_tps:>20.1f}  {vllm_tps:>14.1f}  {ratio:>8.2f}x")
        else:
            print(f"{bs:>6}  {mq_tps:>20.1f}  {'N/A':>14}  {'N/A':>9}")


if __name__ == "__main__":
    main()
