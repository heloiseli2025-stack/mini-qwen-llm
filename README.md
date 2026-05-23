# mini-qwen-llm

A from-scratch high-performance inference engine for Qwen3, built with Triton and PyTorch. Implements the core machinery of a production LLM serving system — paged attention, fused kernels, W4A16 quantization, MoE routing, and continuous batching — targeting Qwen3-8B Dense and Qwen3-30B-A3B MoE on a single GPU.

```
Qwen3-30B-A3B (MoE, 128 experts, top-8)   →  runs on single A100 80GB via W4A16
Qwen3-8B (Dense)                           →  runs on RTX 4090 24GB via W4A16
```

---

## What's inside

| Module | Description | Status |
|--------|-------------|--------|
| **M0** | Qwen3 Dense + MoE model skeleton, GQA + QK-Norm attention, PyTorch baseline | ✅ Done |
| **M1** | Paged Attention — FlashAttention v2 style prefill + decode Triton kernels | ✅ Done |
| **M2** | Fused QKV projection + QK-Norm + RoPE Triton kernel | ✅ Done |
| **M3** | W4A16 GEMM decode kernel, AWQ-style int4 packing/dequant | ✅ Done |
| **M4** | MoE Grouped GEMM — top-K router, permute, fused expert GEMM, unpermute | ✅ Done |
| **M5** | Continuous Batching Scheduler — prefill/decode split, paged KV cache management | ✅ Done |

---

## Performance

### M1 — Paged Attention Decode (RTX 4090 D, H_q=16, H_kv=8, D=128)

Comparison against HF-style naive decode (`torch.cat` + `repeat_interleave` + SDPA):

| Batch | Seqlen | Naive (ms) | Ours (ms) | Speedup |
|------:|-------:|-----------:|----------:|--------:|
| 1 | 2048 | 0.057 | 0.090 | 0.6x |
| 8 | 1024 | 0.224 | 0.087 | **2.6x** |
| 8 | 2048 | 0.528 | 0.093 | **5.7x** |
| 16 | 1024 | 0.536 | 0.088 | **6.1x** |
| 16 | 2048 | 1.108 | 0.164 | **6.7x** |

Speedup grows with sequence length — the naive approach pays O(seq²) memory traffic per step from KV reallocation; paged attention pays O(1) per new token.

### M2 — Fused QKV + QK-Norm + RoPE (RTX 4090 D, Triton 3.4.0)

| Config | Unfused (ms) | Fused (ms) | Speedup |
|--------|-------------|-----------|---------|
| B=4, S=512 | 0.357 | 0.198 | **1.80x** |
| B=1, S=2048 (end-to-end attn) | 1.434 | 0.681 | **2.11x** |
| B=4, S=1024 (end-to-end attn) | 1.662 | 0.827 | **2.01x** |

CUDA kernel launches: 33 → 6 per forward pass. Theoretical occupancy: 100% (26 registers/thread).

### M4 — GPTQ W4A16 MoE Loading

Qwen3-30B-A3B (128 experts, top-8, 48 layers) loaded from GPTQ checkpoint:
- 18,624 quantized Linear layers
- 48 router gates excluded from quantization (dequantized to fp16)
- Top-1 token match verified correct vs. BF16 reference

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              mini-qwen-llm Engine                        │
│                                                          │
│  Scheduler (Continuous Batching, prefill/decode split)   │
│       │                                                  │
│  BlockManager  ←→  KVCache [num_blocks, block_size,      │
│                             num_kv_heads, head_dim]      │
│       │                                                  │
│  ModelRunner                                             │
│    ├─ paged_forward_single_prefill(input_ids, ...)       │
│    └─ paged_forward_decode(input_ids, seq_lens, ...)     │
│         │                                                │
│  Decoder Layer × N                                       │
│    ├─ RMSNorm                                            │
│    ├─ [Triton] Fused QKV + QK-Norm + RoPE               │
│    ├─ [Triton] Paged Attention (prefill / decode)        │
│    ├─ O Projection                                       │
│    ├─ RMSNorm                                            │
│    └─ [Triton] MoE: Router → Permute → GroupedGEMM →    │
│                       Unpermute + Weighted Sum           │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

**Paged Attention** — KV cache is split into fixed-size physical blocks (block_size=16). A `block_table[seq_id, virtual_block_idx]` maps virtual positions to physical blocks, eliminating fragmentation from variable-length sequences. The prefill kernel is FlashAttention v2 style (tiled outer-Q, inner-KV loop); the decode kernel reads K/V indirectly through the block table.

**Fused QKV/RoPE kernel** — Qwen3 adds per-head QK-Norm before RoPE, making the naive approach 5 separate HBM round-trips (QKV proj → split → norm → RoPE → attention). The fused kernel collapses these to 1 read + 1 write per head, using Tensor Core (bf16) for the projection matmul.

**W4A16 GEMM** — Weights are packed as 8×int4 per int32. Dequantization happens in SRAM immediately before `tl.dot`, so we never materialize the full fp16 weight matrix. Group size = 128 (per-group scale + zero point).

**MoE Grouped GEMM** — Naive per-expert for-loops issue 128 separate kernel launches with thin matrices. Instead: (1) Router selects top-K experts per token; (2) Permute kernel physically reorders tokens by expert ID; (3) A single Grouped GEMM processes all experts in one launch, each program indexing its expert weight via `expert_id * weight_stride`; (4) Unpermute kernel reduces weighted outputs back to token order.

**Continuous Batching** — Scheduler runs one step type per iteration: decode (if sequences are running) or prefill (otherwise). Block pre-allocation happens inside `Scheduler.step()` before returning the decode batch, so `ModelRunner.run_decode()` never touches the block manager mid-inference.

**Decode RoPE** — The fused QKV kernel uses `seq_pos = pid_tok % S`, which is only valid when all sequences have the same length. For decode (mixed lengths), the decode path bypasses the fused kernel: unfused QKV projections + QK-Norm + per-sequence RoPE indexed directly from the precomputed `cos_cached / sin_cached` buffers.

---

## Quickstart

### Requirements

- Python 3.10–3.12
- PyTorch ≥ 2.4.0
- Triton ≥ 3.0.0
- CUDA ≥ 12.1, GPU with compute capability ≥ 8.0 (A100 / H100 / RTX 4090)

### Install

```bash
git clone https://github.com/heloiseli2025-stack/mini-qwen-llm
cd mini-qwen-llm
pip install -e .
```

For benchmark comparison against vLLM:
```bash
pip install -e ".[bench]"
```

### Run tests

```bash
# All tests (requires CUDA)
pytest tests/ -v

# CPU-only tests (scheduler logic, no kernels)
pytest tests/test_scheduler.py -v -k "not throughput"

# Specific kernel
pytest tests/test_paged_attention.py -v
pytest tests/test_fused_qkv_rope.py -v
pytest tests/test_w4a16_gemm.py -v
pytest tests/test_moe_kernels.py -v
```

### Download model weights

```bash
# Qwen3-8B Dense (BF16, ~16GB)
python scripts/download_model.py --model Qwen/Qwen3-8B

# Qwen3-30B-A3B MoE (GPTQ Int4, ~18GB)
python scripts/download_model.py --model Qwen/Qwen3-30B-A3B-Instruct-GPTQ-Int4
```

### Run inference

```bash
# Dense model
python scripts/run_inference.py --model weights/Qwen3-8B --prompt "Explain paged attention"

# MoE model (GPTQ)
python scripts/run_inference.py --model weights/Qwen3-30B-A3B-GPTQ --moe --prompt "What is mixture of experts?"
```

### Benchmark throughput

```bash
# Toy model (random weights, no GPU memory needed beyond ~1GB)
python benchmarks/bench_throughput.py --toy --batch 1 4 8 16

# Real model
python benchmarks/bench_throughput.py --model-path weights/Qwen3-8B --batch 1 8 16 32
```

---

## Repository layout

```
mini-qwen-llm/
├── mini_qwen/
│   ├── config.py                    # Qwen3Config, Qwen3MoEConfig
│   ├── model/
│   │   ├── qwen3.py                 # Dense model: forward + paged_forward_*
│   │   ├── qwen3_moe.py             # MoE model: forward + paged_forward_*
│   │   ├── loader.py                # HF safetensors / GPTQ weight loading
│   │   └── layers/
│   │       ├── attention.py         # Qwen3Attention with paged_forward
│   │       ├── rope.py              # RotaryEmbedding + apply_rotary_emb
│   │       ├── moe.py               # Qwen3MoEBlock (router + experts)
│   │       ├── mlp.py               # SwiGLU MLP
│   │       ├── linear_w4a16.py      # W4A16 quantized Linear
│   │       └── rms_norm.py          # RMSNorm
│   ├── kernels/
│   │   ├── paged_attn_prefill.py    # FlashAttn-v2 style prefill + write_kv_decode
│   │   ├── paged_attn_decode.py     # Paged decode kernel (read-only)
│   │   ├── fused_qkv_rope.py        # QKV + QK-Norm + RoPE fusion
│   │   ├── w4a16_gemm.py            # W4A16 decode GEMM
│   │   ├── moe_router.py            # Top-K routing
│   │   ├── moe_permute.py           # Token → expert reordering
│   │   ├── moe_grouped_gemm.py      # Expert batch GEMM
│   │   └── moe_unpermute.py         # Weighted reduction back to tokens
│   ├── cache/
│   │   ├── kv_cache.py              # KVCache [num_blocks, block_size, H_kv, D]
│   │   └── block_manager.py         # Free-list block allocator
│   ├── engine/
│   │   ├── sequence.py              # Sequence state machine
│   │   ├── scheduler.py             # Continuous batching scheduler
│   │   └── runner.py                # ModelRunner + generate_batch loop
│   └── quantization/
│       ├── awq.py                   # AWQ quantization algorithm
│       └── packing.py               # int4 pack/unpack utilities
├── tests/                           # 60 tests, covering all kernels + scheduler
├── benchmarks/                      # bench_attention, bench_w4a16, bench_moe, bench_throughput
├── scripts/                         # download_model, run_inference, compare_with_hf
└── docs/
    ├── 00_baseline.md               # Qwen3 layer shapes and FLOP counts
    ├── 01_paged_attention.md        # Design + benchmark results
    ├── 02_fused_qkv_rope.md         # Design + benchmark results
    ├── 03_w4a16_awq.md              # Design + benchmark results
    ├── 04_moe.md                    # MoE architecture + GPTQ loading
    └── 05_scheduler.md              # Scheduler design decisions
```

---

## Qwen3 architecture notes

Qwen3 differs from Llama-style models in two ways relevant to this engine:

1. **QK-Norm**: Q and K each pass through a per-head RMSNorm *before* RoPE. This stabilizes training but adds an extra HBM round-trip in naive implementations. The fused kernel in M2 absorbs it.

2. **MoE without shared experts**: Qwen3-30B-A3B uses 128 experts with top-8 routing and no shared expert (unlike Qwen2.5-MoE). The router gate produces logits via a simple linear projection; `norm_topk_prob=True` normalizes selected weights to sum to 1.

GQA configuration: Qwen3-8B → 32 Q heads / 8 KV heads (4:1), head_dim=128. Qwen3-30B-A3B → 32 Q heads / 4 KV heads (8:1), head_dim=128.

---

## Known limitations

- Prefill is serial (one sequence at a time). Batched prefill (chunked or varlen) is not implemented.
- No preemption — if KV blocks run out, new requests wait in queue.
- No prefix caching or speculative decoding.
- W4A16 kernel is tuned for decode (small M); prefill (large M) still uses BF16.
- Fused QKV kernel assumes uniform sequence length within a batch. Decode path uses an unfused fallback for per-sequence RoPE positions.

---

## References

- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
- [PagedAttention / vLLM (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [AWQ: Activation-aware Weight Quantization (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)
- [Triton: An Intermediate Language for GPU Programming](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Continuous Batching (Orca, OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
