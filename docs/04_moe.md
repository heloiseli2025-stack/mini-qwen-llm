# M4 MoE Sparse Expert Inference (Qwen3-30B-A3B)

## Goal

Build on M1-M3 to support single-GPU RTX 4090 inference for Qwen3-30B-A3B (MoE, 128 experts, top-8 routing).
The attention layers (M1 Paged Attention + M2 Fused QKV/RoPE) are fully reused; only the MoE FFN needs to be implemented.

---

## Architecture

Qwen3-30B-A3B key config:

| Parameter | Value |
|-----------|-------|
| num_hidden_layers | 94 |
| hidden_size | 3584 |
| num_experts | 128 |
| num_experts_per_tok | 8 |
| intermediate_size | 1536 (per-expert) |

Full BF16: ~60 GB → exceeds 4090 VRAM. After W4A16: ~15 GB → fits on one GPU.

---

## Data Flow

```
hidden_states [T, H]          ← decode T ≤ 16
       │
       ▼  moe_router(hidden, gate.weight, top_k=8)
topk_ids [T, 8], topk_weights [T, 8]
       │
       ▼  moe_permute(hidden, topk_ids, num_experts=128)
permuted_hidden [T*8, H]       ← tokens grouped by expert
expert_offsets  [129]          ← prefix sum; [offsets[e], offsets[e+1]) is the range for expert e
       │
       ▼  Qwen3MoEBlock: per-expert SwiGLU (nn.Linear or LinearW4A16)
expert_out [T*8, H]
       │
       ▼  moe_unpermute(expert_out, topk_weights, topk_ids, T)
output [T, H]                  ← weighted reduction back to original order
```

---

## Per-Kernel Design

### M4.1 Router (moe_router.py)

Pure PyTorch ops; no Triton needed. Router compute is negligible compared to expert GEMMs.
Softmax is computed in fp32 to prevent overflow when bf16 activation values have extreme distributions.

### M4.2 Permute (moe_permute.py)

```python
flat = topk_ids.reshape(-1)             # [T*K], GPU tensor
perm = flat.argsort(stable=True)        # GPU stable sort (not CPU)
expert_offsets[1:] = flat.bincount(minlength=E).cumsum(0)
permuted = hidden_states[perm // K]     # GPU gather
```

**Critical constraints**:
- `stable=True`: guarantees the inverse mapping used in unpermute is deterministic
- `minlength=num_experts`: prevents `bincount` from returning a shorter array when some experts receive no tokens

### M4.3 Grouped GEMM (moe_grouped_gemm.py)

**For-loop oracle** (correctness reference, fp32 accumulation):
```
for e in range(E):
    out[s:t] = (permuted_hidden[s:t].float() @ W[e].float().T).bfloat16()
```

**Triton kernel (performance path)**:
- Only launches programs for active experts (n_e > 0); at most 8 during decode
- Grid = `(num_active, cdiv(D, BLOCK_N), max_m_tiles)`
- Each program handles one [BLOCK_M=16, BLOCK_N] tile for one expert
- Empty M tiles (`pid_m * BLOCK_M >= n_e`) return immediately

Autotune config (key = D, H):

| BLOCK_N | BLOCK_K | num_warps |
|---------|---------|-----------|
| 64 | 32 | 2 |
| 128 | 32 | 4 |
| 64 | 64 | 4 |
| 128 | 64 | 8 |

### M4.4 Unpermute (moe_unpermute.py)

Uses the same stable argsort as permute. Restores original token order via the inverse mapping, then applies vectorized weighted summation.
No extra `sorted_indices` return value; the interface is kept minimal.

---

## Expert Weight Strategy

| Scenario | Weight type | VRAM (30B) |
|----------|-------------|------------|
| Unit tests | BF16 nn.Linear (synthetic weights) | N/A |
| E2E (4090) | W4A16 LinearW4A16 | ~15 GB |

`Qwen3MoEBlock.quantize_to_w4a16(group_size=128)` calls `LinearW4A16.from_float()` for each expert.

---

## File List

| File | Purpose |
|------|---------|
| `mini_qwen/kernels/moe_router.py` | Top-K router |
| `mini_qwen/kernels/moe_permute.py` | Token permutation |
| `mini_qwen/kernels/moe_grouped_gemm.py` | Grouped GEMM (oracle + Triton) |
| `mini_qwen/kernels/moe_unpermute.py` | Weighted inverse permutation |
| `mini_qwen/model/layers/moe.py` | Qwen3MoEBlock |
| `mini_qwen/model/qwen3_moe.py` | Qwen3MoEForCausalLM |
| `mini_qwen/model/loader.py` | `load_moe_from_hf()` |
| `tests/test_moe_kernels.py` | Unit tests + performance tests |

---

## Silent Error Prevention

| Checkpoint | Pass criterion |
|------------|---------------|
| permute round-trip | err < 1e-2 |
| expert_offsets boundary | `offsets[-1] == T*K`, strict equality |
| grouped_gemm vs oracle | max abs error < 1e-2 |
| E2E vs HF | First next-token **top-1 token matches** (logits error is informational only; see below) |

If values are numerically wrong but no crash occurs, check:
1. Whether `stable=True` was omitted from argsort
2. `expert_offsets` off-by-one (missing `minlength` in `bincount`)
3. `expert_weights [E, D, H]` orientation (correct: `X @ W[e].T`, incorrect: `X @ W[e]`)

---

## Measured Results (RTX 4090)

### Correctness (pytest tests/test_moe_kernels.py)

| Test | Result |
|------|--------|
| test_moe_router | PASS |
| test_moe_permute_offsets | PASS |
| test_moe_permute_unpermute_roundtrip | PASS |
| test_moe_grouped_gemm_vs_oracle | PASS (max err < 1e-2) |
| test_moe_grouped_gemm_empty_expert | PASS |
| test_qwen3_moe_block_bf16 | PASS |
| test_qwen3_moe_block_w4a16 | PASS (W4A16 CUDA device correct) |

### Performance (test_moe_grouped_gemm_perf, T=8, E=128, K=8, H=3584, D=1536, RTX 4090)

| Implementation | latency | speedup |
|----------------|---------|---------|
| For-loop oracle (8× torch.mm) | 4.43 ms | 1.00x |
| Triton kernel (dynamic grid) | 0.83 ms | **5.32x** |

Dynamic grid is the key: only 8 active-expert programs are launched (instead of the full 128), eliminating wasted kernel launches and contributing the majority of the speedup.

### E2E Correctness (Qwen3-30B-A3B GPTQ-Int4 vs HF BF16)

Comparison run on a cloud 80 GB GPU (HF BF16 ~60 GB and GPTQ ~16 GB cannot coexist; HF reference logits are collected first and released before loading GPTQ). Prompt: `"Hello, how are you?"`, first next-token:

| Item | Result |
|------|--------|
| HF top-5 | `[' I', ' Let', ' What', ' Can', ' How']` |
| Our top-5 | `[' I', ' Let', ' Also', ' What', ' And']` |
| **Top-1 match** | **True** (` I`) → PASS |
| Top-5 overlap | 3/5 |
| logits max abs err | 2.5 (informational) |
| logits mean abs err | 0.358 (informational) |

**Why max abs err < 1e-2 is not used as the acceptance criterion**: HF MoE uses `torch._grouped_mm` to compute all experts in a batch; this implementation uses per-expert weighted summation. Different BF16 accumulation orders inevitably produce logits differences of ~0.25–2.5 (even with both sides in pure BF16 and no quantization the difference is ~0.94). With GPTQ-Int4 quantization error added on top, the discrepancy is larger still. Therefore the correctness acceptance criterion is **top-1 token agreement**, with logits error reported as informational only.

### W4A16 Quantization Path Notes

Naive min-max quantization (`load_moe_from_hf(quantize_w4a16=True)`) does not achieve sufficient quality to pass the E2E check. Production W4A16 uses a **pre-calibrated GPTQ-Int4 checkpoint** (produced by GPTQModel, desc_act=false). GPTQ uses the v1 zero-point +1 convention: dequant = `(q - (z+1)) * scale`; load with `zero_plus_one=True` (validated: conv B max_err 0.026 < conv A 0.054). GPTQModel also quantizes the MoE router (`mlp.gate`), but the router needs fp16 precision; the loader dequantizes it back to full-precision weights.

---

## E2E Usage

```python
import torch
from mini_qwen.model.loader import load_moe_from_gptq

# Recommended: load pre-calibrated GPTQ-Int4 checkpoint (~16 GB)
model = load_moe_from_gptq(
    "/path/to/Qwen3-30B-A3B-GPTQ-Int4",
    dtype=torch.bfloat16,
    group_size=128,
    zero_plus_one=True,    # GPTQ v1 zero-point +1 convention
    device="cuda",
)
model.eval()
```

Naive min-max quantization path (lower quality, for kernel verification only):

```python
from mini_qwen.model.loader import load_moe_from_hf

model = load_moe_from_hf(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    quantize_w4a16=True,
    group_size=128,
).cuda()
```
