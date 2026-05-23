# M1 Paged Attention

## Why Paged Attention

The naive approach to KV cache management in autoregressive decoding is to concatenate new K/V tensors to the existing cache at each step:

```
k_cache = torch.cat([k_cache, new_k], dim=2)   # O(seq) copy every step
v_cache = torch.cat([v_cache, new_v], dim=2)
```

This has two critical problems:

1. **O(seq²) memory traffic**: At decode step t, copying the full KV history of length t means total memory traffic grows quadratically with sequence length. For a batch of 16 sequences at length 2048 this becomes the dominant cost.

2. **Memory fragmentation**: Each sequence holds a contiguous allocation that must grow. Pre-allocating a worst-case buffer wastes memory; growing it requires reallocation and copying. With dynamic batch sizes and variable sequence lengths, GPU memory becomes badly fragmented.

Paged Attention solves both problems by storing KV cache in fixed-size **blocks** (block_size=16 tokens per block). A `block_table[seq_id]` maps logical block indices to physical block slots in a pre-allocated pool. New tokens fill the next available block; no copying is needed. Physical memory is non-contiguous but the kernel resolves the mapping at runtime.

## Design

### block_table layout

```
block_table: int32[max_seqs, max_blocks_per_seq]
   block_table[s, b] = physical block index for sequence s, logical block b

kv_pool: bf16[num_blocks, 2, H_kv, block_size, D]
   kv_pool[blk, 0]  = K blocks
   kv_pool[blk, 1]  = V blocks
```

The decode step allocates a new physical block when `seq_len % block_size == 0`, updating `block_table` in the scheduler before the kernel runs. No block allocation happens inside the kernel itself.

### Two-kernel approach

**Prefill** (`paged_attn_prefill`): A Triton tiled causal attention kernel that simultaneously writes Q*K^T + softmax + AV and fills the KV cache blocks. Grid shape covers all (batch, head, query-tile) combinations. The causal mask is applied within each tile.

**Decode** (`paged_attn_decode`): A single-pass kernel per query token. For each (batch, head) pair, the kernel walks the `block_table`, loads K/V blocks one at a time, and accumulates the attention output using online softmax. GQA is handled by broadcasting: multiple Q heads map to the same KV head via `pid_kv_head = pid_q_head // (H_Q // H_KV)`.

### block_size=16

block_size=16 was chosen to balance:
- **Register pressure**: Loading one block at a time keeps register usage low.
- **Overhead**: Smaller blocks increase `block_table` lookup frequency; 16 tokens/block is a reasonable middle ground.
- **Alignment**: 16 × D=128 = 2048 elements per block, fitting cleanly in shared memory.

### Implementation notes

**Online softmax** (decode kernel): The standard two-pass softmax is replaced by a running-max + running-sum accumulator updated block by block, following the Flash Attention numerically stable formulation. This allows single-pass traversal without storing the full attention score vector.

**GQA broadcast**: `H_Q=16`, `H_KV=8`, ratio=2. Each K/V head serves two Q heads. The kernel index `pid_kv_head = pid_q_head // 2` selects the correct K/V head without materializing expanded K/V tensors.

**write_kv_decode separation**: During decode the KV write (`write_kv_decode`) is a separate lightweight kernel that stores the single new token's K/V into the correct block slot before the attention kernel reads the full history. This avoids a RAW hazard and keeps the attention kernel read-only with respect to the pool.

## Benchmark Results

**GPU**: NVIDIA GeForce RTX 4090 D
**PyTorch**: 2.8.0+cu128
**Date**: 2026-05-20
**Config**: H_q=16, H_kv=8, D=128, block_size=16
**Timing**: decode warmup=10/reps=100; prefill warmup=5/reps=30

### system_bench (decode)

baseline = `torch.cat([k_prev, new_k], dim=2)` + `repeat_interleave` + SDPA
Simulates HF-style native decode: full KV copy every step, O(seq²) memory traffic.

| batch | seqlen | baseline (ms) | ours (ms) | speedup |
|------:|-------:|--------------:|----------:|--------:|
| 1 | 512 | 0.055 | 0.088 | **0.6x** |
| 1 | 1024 | 0.056 | 0.087 | **0.6x** |
| 1 | 2048 | 0.057 | 0.090 | **0.6x** |
| 8 | 512 | 0.082 | 0.087 | **0.9x** |
| 8 | 1024 | 0.224 | 0.087 | **2.6x** |
| 8 | 2048 | 0.528 | 0.093 | **5.7x** |
| 16 | 512 | 0.249 | 0.088 | **2.8x** |
| 16 | 1024 | 0.536 | 0.088 | **6.1x** |
| 16 | 2048 | 1.108 | 0.164 | **6.7x** |

### kernel_bench (decode)

baseline = contiguous KV (single allocation, no cat) + GQA pre-expand + SDPA
Measures attention compute only, excluding KV concatenation overhead.

| batch | seqlen | baseline (ms) | ours (ms) | speedup |
|------:|-------:|--------------:|----------:|--------:|
| 1 | 512 | 0.019 | 0.088 | **0.2x** |
| 1 | 1024 | 0.019 | 0.087 | **0.2x** |
| 1 | 2048 | 0.022 | 0.090 | **0.2x** |
| 8 | 512 | 0.031 | 0.087 | **0.4x** |
| 8 | 1024 | 0.049 | 0.087 | **0.6x** |
| 8 | 2048 | 0.150 | 0.093 | **1.6x** |
| 16 | 512 | 0.084 | 0.087 | **1.0x** |
| 16 | 1024 | 0.185 | 0.088 | **2.1x** |
| 16 | 2048 | 0.362 | 0.165 | **2.2x** |

### prefill_bench

baseline 1 (naive) = `sdpa_kernel(MATH)` + GQA pre-expand, O(n²) memory, no Flash Attention
baseline 2 (fa2)   = `sdpa_kernel(FLASH_ATTENTION)` + GQA pre-expand (PyTorch 2.8 dense SDPA requires equal num_heads)
ours = `paged_attn_prefill` (Triton tiled causal, **includes KV cache write overhead**; both baselines exclude this)

| batch | seqlen | naive (ms) | fa2 (ms) | ours (ms) | vs naive | vs fa2 |
|------:|-------:|-----------:|---------:|---------:|---------:|-------:|
| 1 | 512 | 0.192 | 0.039 | 0.298 | **0.64x** | **0.13x** |
| 1 | 1024 | 0.897 | 0.056 | 0.306 | **2.93x** | **0.18x** |
| 1 | 2048 | 3.719 | 0.179 | 0.449 | **8.28x** | **0.40x** |
| 1 | 4096 | 15.185 | 0.593 | 0.987 | **15.38x** | **0.60x** |
| 4 | 512 | 1.046 | 0.051 | 0.299 | **3.50x** | **0.17x** |
| 4 | 1024 | 4.057 | 0.157 | 0.447 | **9.07x** | **0.35x** |
| 4 | 2048 | 14.781 | 0.556 | 1.026 | **14.41x** | **0.54x** |
| 4 | 4096 | 60.541 | 2.045 | 3.176 | **19.07x** | **0.64x** |

## Key Observations

### Decode (system_bench vs kernel_bench)

- **system_bench** speedup range: 0.6x (batch=1, seq=512) to 6.7x (batch=16, seq=2048). This includes `torch.cat` memory allocation overhead, which grows with sequence length — explaining why the gap widens at longer sequences.

- **kernel_bench** speedup range: 0.2x (batch=1, seq=512) to 2.2x (batch=16, seq=2048). When KV concatenation is excluded, our kernel is slower at small sequence lengths because block_table indirect addressing introduces cache misses that SDPA on contiguous memory avoids. At large sequence lengths (batch=16, seq=2048) paged attention wins because it avoids allocating large contiguous KV blocks.

- **batch=1 is always slower**: With only one sequence the block_table walk adds overhead that is not amortized. Paged attention's advantage is a batch-level phenomenon.

### Prefill

- **vs naive O(n²)**: 0.64x to 19.07x speedup. The naive SDPA materializes the full N×N attention matrix; our Triton tiled kernel is strictly better at long sequences.

- **vs FA2**: 0.13x to 0.64x (< 1 means our kernel is slower). This gap is expected: our prefill kernel writes KV cache blocks during the forward pass, which neither baseline does. FA2 is also a heavily optimized CUDA kernel; the Triton implementation cannot match it at small batch sizes.

- **Engineering trade-off**: The paged prefill kernel supports dynamic block allocation and prefix cache sharing. The cost relative to FA2 is the price of that flexibility.
