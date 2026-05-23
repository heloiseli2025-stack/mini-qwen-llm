# M2 Fused QKV + QK-Norm + RoPE

## Motivation

Qwen3's attention front-half adds a per-head QK-RMSNorm step that Llama does not have. The unfused execution path looks like:

```
x → GEMM_Q → (HBM write) → RMSNorm_Q → (HBM write) → RoPE_Q → (HBM write)
x → GEMM_K → (HBM write) → RMSNorm_K → (HBM write) → RoPE_K → (HBM write)
x → GEMM_V → (HBM write)
```

Q and K each require 5 HBM operations, totalling 10+ round-trips (excluding V).

## Implementation Strategy

| Step | Implementation | Rationale |
|------|----------------|-----------|
| QKV GEMM | `torch.mm` (cuBLAS) | cuBLAS is far faster than Triton for large matmuls |
| QK-Norm + RoPE | Single Triton kernel | Eliminates intermediate HBM write-backs |
| V | GEMM output used directly | V requires no norm or RoPE |

After fusion, Q and K each require 3 HBM operations (GEMM write + kernel read + kernel write), saving 40%.

## Kernel Design

**File**: `mini_qwen/kernels/fused_qkv_rope.py`

**Grid**: `(B×S, H_Q + H_KV)` — a single kernel launch covers all Q and K heads

```
pid_head 0..H_Q-1          → Q head (is_q = True)
pid_head H_Q..H_Q+H_KV-1  → K head (is_q = False)
```

**Each program handles 1 (token, head) pair**:
1. Load in two halves (HALF_D = 64 elements, avoids register-side gather)
2. RMSNorm: fp32 accumulation (`tl.sum(x1*x1) + tl.sum(x2*x2)`), preventing bf16 precision loss
3. RoPE: `out1 = x1_n * cos1 - x2_n * sin1`, `out2 = x2_n * cos2 + x1_n * sin2`
4. Cast to bf16 and write back

**RoPE formula verification** (corresponds to `rotate_half`):
```
rotate_half([x1, x2]) = [-x2, x1]
q_rope = q * cos + rotate_half(q) * sin
       = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]
       = [x1*cos1 - x2*sin1, x2*cos2 + x1*sin2]
```

## Correctness and Performance Results

**Correctness** (RTX 4090 D, Triton 3.4.0):

| Test | Result |
|------|--------|
| Q max abs error | ≤ 0.0078 < 1e-2 ✓ |
| K max abs error | ≤ 0.0078 < 1e-2 ✓ |
| V passthrough error | 0.0 ✓ |

**Performance** (B=4, S=512, warmup=10, reps=100, CUDA events):

| Metric | unfused | fused | improvement |
|--------|---------|-------|-------------|
| Latency | 0.357 ms | 0.198 ms | **1.80x** |
| CUDA kernel count | 33 | **6** | 5.5x reduction |

Fused kernel breakdown: 3× cuBLAS GEMM + 1× `_qk_norm_rope_kernel` + `aten::view` and other meta-ops = 6 CUDA events total.

## End-to-End Attention Latency

Measured scope: QKV projection + QK-Norm + RoPE + Paged Prefill (`paged_attn_prefill`).
Excludes o_proj. warmup=10, reps=100.

| B | S | unfused (ms) | fused (ms) | speedup |
|---|---|-------------|-----------|---------|
| 1 | 512  | 0.762 | 0.532 | 1.43x |
| 1 | 1024 | 0.763 | 0.532 | 1.43x |
| 1 | 2048 | 1.434 | 0.681 | **2.11x** |
| 4 | 512  | 0.927 | 0.532 | 1.74x |
| 4 | 1024 | 1.662 | 0.827 | 2.01x |
| 4 | 2048 | 2.782 | 1.708 | 1.63x |

Speedup range: **1.43x to 2.11x**. Longer sequences benefit more because the norm+RoPE HBM traffic scales linearly with token count, while the Flash Attention portion is unaffected by the QKV path.

## Occupancy Analysis

> Cloud environment (AutoDL) has `RmProfilingAdminOnly=1`; ncu performance counters are restricted.
> The figures below are derived from Triton's compiled kernel metadata (`CompiledKernel.metadata`) and the CUDA occupancy formula.

**Compiled kernel info** (Triton 3.4.0, SM 8.9, RTX 4090 D):

| Metric | Value |
|--------|-------|
| `n_regs` / thread | 26 |
| `num_warps` / block | 4 (= 128 threads/block) |
| Shared memory / block | 8 bytes |
| Spills | 0 |

**Occupancy estimate** (RTX 4090 D, SM 8.9, max 48 warps/SM, 65536 regs/SM):

| Limiting factor | Calculation | max blocks/SM |
|-----------------|-------------|--------------|
| Warp count | 48 / 4 | **12** |
| Thread count | 1536 / 128 | 12 |
| Register usage | ⌊65536 / (26×128)⌋ = 19 | 19 |
| Shared memory | 102400 / 8 = 12800 | 12800 |

Bottleneck: warp count, max 12 blocks/SM × 4 warps = **48 warps/SM**

```
Theoretical Occupancy    : 100%  (48/48 warps)
Active Warps Per SM      : 48
Active Threads Per SM    : 1536
```

> Estimated achieved occupancy >= 80%: grid size 2048×24 = 49152 programs is far larger than
> 114 SM × 12 blocks = 1368 concurrent blocks, so SMs can stay fully occupied. The kernel has
> no `__syncthreads`; all programs are fully independent with no warp divergence bottleneck.
>
> Precise achieved occupancy requires `ncu --section Occupancy`, available with
> `/usr/local/cuda/bin/ncu` in environments that allow performance counters.

**Conclusion**: 26 regs/thread is extremely low (10% of the 255-register cap). Registers are not the bottleneck; there is no benefit to splitting into separate Q/K kernels. The current single-kernel design already achieves maximum theoretical occupancy — no further optimization is possible on this axis.

## HBM Traffic Comparison

| Configuration | Q+K HBM ops | Notes |
|---------------|-------------|-------|
| Unfused | 10 | GEMM write×2 + norm read×2 + norm write×2 + RoPE read×2 + RoPE write×2 |
| Fused | 6 | GEMM write×2 + kernel read×2 + kernel write×2 |
| Savings | 4 (-40%) | |

## Roofline Analysis

The `_qk_norm_rope_kernel` has very low arithmetic intensity (pure vector operations: RMSNorm + RoPE, no matrix multiply). The optimization goal is to reduce HBM traffic rather than to increase Tensor Core utilization. The kernel operates in the **memory-bound regime**. Theoretical occupancy of 100% indicates the SM can fully hide memory latency. Register usage of only 26/thread (10% of the 255-register cap) with zero register spills is the ideal profile for a memory-bound kernel.

> TODO: In an environment where ncu is not restricted (non-containerized instance), add measured
> `dram__bytes_read.sum` / `dram__bytes_write.sum` values to confirm how closely the actual HBM
> savings match the theoretical -40%.
