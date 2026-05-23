"""End-to-end token/s throughput benchmark (M5 implementation).

Benchmark protocol (§4.7, frozen):
    - Warmup: 20 iters (not counted)
    - Measurement: 100 iters
    - Report: median tok/s, p25, p75
    - Default: prompt_len=2048, gen_len=256, batch=[1,8,16,32]

Usage:
    # Toy model (random weights, fast functional check)
    python benchmarks/bench_throughput.py --toy

    # Real GPTQ model
    python benchmarks/bench_throughput.py --model-path weights/Qwen3-30B-A3B-GPTQ-Int4

    # Custom batch / length
    python benchmarks/bench_throughput.py --model-path ... --batch 1 8 16 --prompt-len 512 --gen-len 64
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

import torch

from mini_qwen.cache.block_manager import BlockManager
from mini_qwen.cache.kv_cache import KVCache, KVCacheConfig
from mini_qwen.engine.runner import ModelRunner, generate_batch
from mini_qwen.engine.scheduler import Scheduler


def _make_toy_model_and_caches(device: torch.device):
    """2-layer random-weight model for functional verification (no GPU memory needed)."""
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.qwen3_moe import Qwen3MoEForCausalLM

    cfg = Qwen3MoEConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    model = Qwen3MoEForCausalLM(cfg).to(device).eval()
    num_blocks = 1024
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        block_size=16,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    kv_caches = [KVCache(kv_cfg) for _ in range(cfg.num_hidden_layers)]
    return model, kv_caches, num_blocks, cfg.num_key_value_heads, cfg.head_dim


def _make_real_model_and_caches(model_path: str, device: torch.device):
    """Load a real GPTQ checkpoint."""
    from mini_qwen.model.loader import load_moe_from_gptq

    model = load_moe_from_gptq(model_path, device=device)
    model.eval()
    cfg = model.config
    num_blocks = 2048
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        block_size=16,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    kv_caches = [KVCache(kv_cfg) for _ in range(cfg.num_hidden_layers)]
    return model, kv_caches, num_blocks, cfg.num_key_value_heads, cfg.head_dim


def bench_one(
    model,
    kv_caches: list[KVCache],
    num_blocks: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup: int,
    measure: int,
    device: torch.device,
) -> dict:
    """Run warmup + measure rounds for a given batch_size; return stats."""
    base = list(range(512))
    prompt = (base * ((prompt_len // 512) + 1))[:prompt_len]
    prompts = [prompt[:] for _ in range(batch_size)]

    def _reset_kv():
        for kv in kv_caches:
            kv.k_cache.zero_()
            kv.v_cache.zero_()

    def _one_iter() -> float:
        _reset_kv()
        bm = BlockManager(num_blocks=num_blocks, block_size=16)
        sched = Scheduler(bm, max_seqs_in_flight=batch_size + 4)
        runner = ModelRunner(model, kv_caches, bm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = generate_batch(runner, sched, prompts, max_new_tokens=gen_len)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_toks = sum(len(v) for v in outs.values())
        return total_toks / elapsed   # tok/s

    for _ in range(warmup):
        _one_iter()

    tps_list = [_one_iter() for _ in range(measure)]
    med = statistics.median(tps_list)
    p25 = sorted(tps_list)[measure // 4]
    p75 = sorted(tps_list)[measure * 3 // 4]
    return {"batch": batch_size, "median_tps": med, "p25": p25, "p75": p75}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to GPTQ checkpoint; omit to use toy random-weight model")
    parser.add_argument("--toy", action="store_true",
                        help="Force toy random-weight model (overrides --model-path)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")

    if args.toy or args.model_path is None:
        print("Using toy random-weight model (functional verification mode)")
        args.prompt_len = min(args.prompt_len, 64)
        args.gen_len    = min(args.gen_len, 8)
        args.warmup     = 2
        args.measure    = 5
        model, kv_caches, num_blocks, _, _ = _make_toy_model_and_caches(device)
    else:
        print(f"Loading GPTQ checkpoint: {args.model_path}")
        model, kv_caches, num_blocks, _, _ = _make_real_model_and_caches(args.model_path, device)

    print(f"\n{'batch':>6}  {'median tok/s':>14}  {'p25':>10}  {'p75':>10}")
    print("-" * 46)
    for bs in args.batch:
        result = bench_one(
            model, kv_caches, num_blocks, bs,
            args.prompt_len, args.gen_len,
            args.warmup, args.measure, device,
        )
        print(f"{result['batch']:>6}  {result['median_tps']:>14.1f}  "
              f"{result['p25']:>10.1f}  {result['p75']:>10.1f}")


if __name__ == "__main__":
    main()
