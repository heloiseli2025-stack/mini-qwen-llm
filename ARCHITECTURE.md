# Architecture & Technical Specification

This document describes the design rationale, kernel specifications, frozen interfaces, and benchmark protocols for mini-qwen-llm. It is intended for contributors who want to understand why things are built the way they are, or who are extending the engine.

---

## 1. Model choice: why Qwen3

| Dimension | Qwen3-8B Dense | Qwen3-30B-A3B MoE |
|-----------|---------------|-------------------|
| Architecture | GQA + **QK-Norm** | **MoE** (128 experts / top-8) |
| Single-GPU (24GB) | ✅ BF16 tight, W4A16 comfortable | ❌ BF16 OOM; ✅ W4A16 ~15GB |
| Interesting kernel work | QK-Norm fusion, GQA | Expert routing, Grouped GEMM |
| Long context | 128K | 128K |

Qwen3 has two architectural differences from Llama that require new kernel logic:

- **QK-Norm**: per-head RMSNorm on Q and K before RoPE. Naive implementation adds an HBM round-trip; the fused kernel (M2) eliminates it.
- **No shared expert**: Qwen3-30B-A3B uses 128 routed experts with no shared expert path (unlike Qwen2.5-MoE), simplifying the MoE forward but making load balancing purely training-side.

---

## 2. Frozen interfaces

These data structures and function signatures are stable. Changing them requires updating all downstream consumers simultaneously.

### 2.1 KV Cache physical layout

```python
@dataclass(frozen=True)
class KVCacheConfig:
    num_blocks: int
    block_size: int = 16      # fixed — matches vLLM default
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16

# K cache: [num_blocks, block_size, num_kv_heads, head_dim]
# V cache: [num_blocks, block_size, num_kv_heads, head_dim]
# Layout rationale: block_size before num_kv_heads ensures tokens within
# a block are contiguous, which is what the decode kernel needs for
# efficient block-table-indexed loads.
```

### 2.2 Block table

```python
# Shape: [max_num_seqs, max_blocks_per_seq], dtype=int32
# block_table[seq_id, virtual_block_idx] = physical_block_id
# -1 means unallocated
```

### 2.3 Sequence state machine

```python
@dataclass
class Sequence:
    seq_id: int                   # immutable
    prompt_token_ids: list[int]   # immutable
    output_token_ids: list[int]   # append-only
    block_ids: list[int]          # append-only
    status: Literal["waiting", "running", "finished"]
```

### 2.4 Public kernel signatures

Each kernel file has a `FROZEN SIGNATURE` comment block at the top. The internal implementation may be optimized freely, but the Python-visible signatures are stable.

---

## 3. Module specifications

### M1 — Paged Attention

**Goal**: eliminate KV cache memory fragmentation; support dynamic batching of variable-length sequences.

**Two kernels**:

*Prefill* (`paged_attn_prefill`):
- Input: Q/K/V packed `[total_tokens, H, D]` bf16, `cu_seqlens [B+1]`, `block_table [B, max_blocks]`
- Algorithm: FlashAttention v2 tiled causal attention (outer Q tiles, inner KV tiles), writes K/V to paged cache as a side effect
- Grid: `(ceil(max_seqlen / BLOCK_Q), batch, H_q)`
- Accumulation in fp32; output cast to bf16

*Decode* (`paged_attn_decode`):
- Input: Q `[B, H_q, D]` bf16, populated K/V cache, `block_table [B, max_blocks]`, `seq_lens [B]`
- Algorithm: per-(seq, head) program; loads K/V tiles through `block_table` indirect addressing; online softmax with running `(m_i, l_i)`
- Read-only: does **not** write new K/V (that is done by `write_kv_decode` in `paged_attn_prefill.py`)
- Grid: `(B, H_q)`

**GQA**: `num_kv_groups = H_q // H_kv`; each KV head is shared by `num_kv_groups` Q heads. The decode kernel computes `kv_head_id = head_id // num_kv_groups`.

**Block table access pattern**:
```
pos = token position in sequence
page_idx = pos // PAGE_SIZE
slot     = pos  % PAGE_SIZE
phys     = block_table[seq_id, page_idx]
K_cache[phys, slot, head_id, :]
```

**Write on decode**: before calling `paged_attn_decode`, the new token's K/V must be written to cache. This is done by `write_kv_decode`, which reuses `_write_kv_cache_kernel` with `batch_ids = arange(B)` and `positions = seq_lens - 1`.

### M2 — Fused QKV + QK-Norm + RoPE

**Goal**: collapse 5 sequential HBM round-trips (QKV matmul → split → QK-Norm → RoPE → attention) into 1 read + 1 write per head.

**Kernel structure**: one program per `(token, head_group)`. Each program:
1. Loads the token's hidden state tile
2. Computes Q/K/V via `tl.dot` (Tensor Core, bf16)
3. Applies per-head RMSNorm to Q and K (V is unchanged)
4. Applies RoPE: `q_rot = q * cos[pos] + rotate_half(q) * sin[pos]`
5. Stores Q/K/V

**Limitation**: uses `seq_pos = pid_tok % S`, which is only valid when all sequences in the batch have the same length. The decode path (where each sequence has a different RoPE position) uses an unfused fallback.

**QK-Norm**: Qwen3-specific. Weight shape `[head_dim]`, applied per head. The fused kernel loads `q_norm.weight` and `k_norm.weight` as constexpr-addressed buffers.

### M3 — W4A16 GEMM

**Goal**: reduce weight memory bandwidth by 4× for all linear layers.

**Packing**: 8 int4 values packed into one int32. Packing order is low-bit-first: `packed[i//8] |= (val & 0xF) << ((i%8)*4)`. Unpack reverses this exactly.

**Dequantization** (in SRAM, not pre-materialized):
```
w_fp16 = (w_int4 - zero_point) * scale
```
Group size = 128: each 128 K-dimension elements share one (scale, zero_point) pair.

**Kernel structure**: one program per `[BLOCK_M, BLOCK_N]` output tile. Loads the packed weight tile, unpacks to bf16 in registers, calls `tl.dot` for the GEMM accumulation.

**GPTQ loading**: `loader.py` handles `zero_plus_one=True` convention (GPTQ stores `qzeros + 1` so that the zero point for all-zero packed weights is correct). Router gates are excluded from quantization and dequantized to fp16.

### M4 — MoE Grouped GEMM

**Goal**: replace 128 sequential expert GEMMs with a single batched operation.

**Four-stage pipeline**:

1. **Router** (`moe_router.py`): computes `gate_logits = hidden @ router_weight`, takes top-K, applies softmax. Output: `topk_ids [T, K]`, `topk_weights [T, K]`.

2. **Permute** (`moe_permute.py`): reorders tokens so that all tokens routed to expert `e` are contiguous. Computes `expert_offsets [num_experts+1]` (prefix sum of expert token counts). Output: `permuted_hidden [T*K, D]`.

3. **Grouped GEMM** (`moe_grouped_gemm.py`): one program per `(expert, BLOCK_M tile, BLOCK_N tile)`. Each program finds its expert's weight via `expert_id * weight_stride` and its token slice via `expert_offsets`. Supports W4A16 weights.

4. **Unpermute + reduce** (`moe_unpermute.py`): for each original token, sums the `K` expert outputs weighted by `topk_weights`. Output: `[T, D]`.

**Why permute?** The alternative is per-expert indexing with `tl.gather` inside the GEMM. Permute trades one extra kernel launch for contiguous memory access in the GEMM, which is critical for Tensor Core utilization.

### M5 — Continuous Batching Scheduler

**Goal**: eliminate the "head-of-line blocking" in static batching where short sequences wait for the longest one in the batch.

**Scheduling policy**:
- If `running` is non-empty → decode (all running sequences advance one token)
- If `running` is empty → prefill (take from `waiting` up to `max_seqs_in_flight` and `max_prefill_tokens` budget)
- Prefill is currently single-sequence (B=1) due to the fused QKV kernel's same-length-sequence assumption.

**Block pre-allocation**: happens in `Scheduler.step()` before returning the decode batch. When `(total_len - 1) % block_size == 0`, the next decode step writes to the first slot of a new page. The block is allocated (or skipped if OOM) before `ModelRunner.run_decode()` is called.

**Decode position tracking**:
- `seq_lens_new[i] = seq.total_len` (prompt + output tokens already appended)
- New K/V is written at position `seq_lens_new - 1`
- `paged_attn_decode` attends over `seq_lens_new` tokens (0 through `seq_lens_new - 1`)

**OOM policy**: no preemption. If blocks are insufficient for a new prefill request, it stays in `waiting`. If a decode step's block pre-allocation fails (rare), the sequence reuses its last page slot on the next step.

---

## 4. Benchmark protocol

All benchmark numbers must be collected under these fixed conditions.

### Hardware and software

- GPU: single-device, no other GPU workload during measurement
- CUDA ≥ 12.1, PyTorch ≥ 2.4.0, Triton ≥ 3.0.0
- All kernels warmed up before timing

### Timing config

| Parameter | Value |
|-----------|-------|
| Warmup | 20 iterations (not counted) |
| Measurement | 100 iterations |
| Statistic | **Median** (not mean — avoids outlier inflation) |
| Synchronization | `torch.cuda.synchronize()` before and after each timed block |

### Decode throughput benchmark

- Prompt length: **2048** (fixed)
- Generation length: 256
- Batch sizes: {1, 8, 16, 32}
- Metric: `tokens/s = (gen_len × batch_size) / median_wall_time`

### Prefill latency benchmark

- Prompt lengths: {128, 512, 2048, 8192}
- Batch: 1
- Metric: time-to-first-token (ms)

### End-to-end throughput benchmark

- 100 requests, `prompt_len ~ Uniform(128, 2048)`, `gen_len = 256`
- Metric: total tokens / total wall time; also P50 / P99 latency

### Required reporting fields

1. Full command with all arguments
2. Git commit hash
3. Peak GPU memory (`torch.cuda.max_memory_allocated()`)
4. Comparison table against HF and/or vLLM at the same batch/seqlen/dtype

---

## 5. Numerical correctness tolerances

| dtype | rtol | atol | Used for |
|-------|------|------|----------|
| fp32 | 1e-5 | 1e-5 | Oracle-level verification |
| bf16 | 1e-2 | 1e-2 | All Triton kernel outputs |

Every Triton kernel must have a corresponding PyTorch reference implementation that serves as a numerical oracle. The oracle is kept in the test file; it does not need to be fast.

---

## 6. Common Triton debugging patterns

### Silent numerical error

1. Reduce to minimum: `BLOCK_SIZE=1`, single head, fp32 accumulation, batch=1, seqlen=16
2. Compare with PyTorch oracle layer by layer — find the first divergence point
3. Check strides: print `tensor.stride()` and verify the kernel's pointer arithmetic matches
4. Check mask: `tl.load(ptr, mask=..., other=0.0)` — boundary conditions and the `other` fill value
5. Check transpose: is `tl.dot(A, B)` expecting `(M,K) @ (K,N)` or `(M,K) @ (N,K)`?

### NaN

1. Online softmax: confirm `m_i` is updated before `l_i` rescaling; confirm exp receives `score - m_new`, not `score`
2. Softmax denominator: `l_i` can be zero if all scores are `-inf` (full mask)
3. RMSNorm: `eps=1e-6` may underflow in bf16 — use `eps=1e-5`

### W4A16 PPL explosion

1. Packing direction: `(qweight >> (i * 4)) & 0xF` vs `(qweight >> ((7-i) * 4)) & 0xF` — must match how the weights were packed
2. Zero point convention: AWQ uses `(w_int4 - zero_point) * scale`; GPTQ `zero_plus_one=True` stores `zero_point + 1`
3. Scale broadcast direction: `scales [K/group_size, N]` — verify the K dimension is the group dimension

### Performance below target

1. `TRITON_PRINT_AUTOTUNING=1` — see which config was selected
2. `ncu --section LaunchStats --section Occupancy` — check register pressure and occupancy
3. Verify `tl.dot` is actually hitting Tensor Core (requires bf16 or fp16 inputs)
4. Check register spill: >64 regs/thread typically causes spill → 3–5× latency increase
