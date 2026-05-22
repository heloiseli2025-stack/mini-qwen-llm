# M4 MoE 稀疏 Expert 推理（Qwen3-30B-A3B）

## 目标

在 M1-M3 基础上支持 Qwen3-30B-A3B（MoE，128 expert，top-8 routing）单卡 4090 推理。
Attention 层（M1 Paged Attn + M2 Fused QKV/RoPE）完全复用，只需实现 MoE FFN 部分。

---

## Architecture

Qwen3-30B-A3B 关键 config：

| 参数 | 值 |
|------|----|
| num_hidden_layers | 94 |
| hidden_size | 3584 |
| num_experts | 128 |
| num_experts_per_tok | 8 |
| intermediate_size | 1536（per-expert） |

BF16 全量：约 60 GB → 超出 4090 显存。W4A16 后约 15 GB → 可运行。

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
expert_offsets  [129]          ← prefix sum，[offsets[e], offsets[e+1]) 是 expert e 的 range
       │
       ▼  Qwen3MoEBlock：per-expert SwiGLU（nn.Linear 或 LinearW4A16）
expert_out [T*8, H]
       │
       ▼  moe_unpermute(expert_out, topk_weights, topk_ids, T)
output [T, H]                  ← 加权还原
```

---

## 各 Kernel 设计

### M4.1 Router（moe_router.py）

纯 PyTorch ops，无需 Triton。Router 计算量相比 Expert GEMM 可忽略。
softmax 在 fp32 做，防止 bf16 激活值分布极端时的 overflow。

### M4.2 Permute（moe_permute.py）

```python
flat = topk_ids.reshape(-1)             # [T*K]，GPU tensor
perm = flat.argsort(stable=True)        # GPU stable sort（非 CPU）
expert_offsets[1:] = flat.bincount(minlength=E).cumsum(0)
permuted = hidden_states[perm // K]     # GPU gather
```

**关键约束**：
- `stable=True`：保证 unpermute 的逆映射确定性
- `minlength=num_experts`：防止未被选中的 expert 导致 bincount 长度不足

### M4.3 Grouped GEMM（moe_grouped_gemm.py）

**for-loop oracle**（正确性 reference，fp32 accumulation）：
```
for e in range(E):
    out[s:t] = (permuted_hidden[s:t].float() @ W[e].float().T).bfloat16()
```

**Triton kernel（性能版）**：
- 只对 active experts（n_e > 0）启动程序，decode 时最多 8 个
- Grid = `(num_active, cdiv(D, BLOCK_N), max_m_tiles)`
- 每个 program：处理一个 expert 的一个 [BLOCK_M=16, BLOCK_N] tile
- 空 M tile（`pid_m * BLOCK_M >= n_e`）直接 return

Autotune 配置（key = D, H）：

| BLOCK_N | BLOCK_K | num_warps |
|---------|---------|-----------|
| 64 | 32 | 2 |
| 128 | 32 | 4 |
| 64 | 64 | 4 |
| 128 | 64 | 8 |

### M4.4 Unpermute（moe_unpermute.py）

与 permute 使用相同的 stable argsort，通过逆映射还原顺序，向量化加权求和。
无额外 sorted_indices 返回值，接口简洁。

---

## Expert 权重策略

| 场景 | 权重类型 | 显存（30B） |
|------|---------|------------|
| 单元测试 | BF16 nn.Linear（合成权重） | N/A |
| E2E（4090） | W4A16 LinearW4A16 | ~15 GB |

`Qwen3MoEBlock.quantize_to_w4a16(group_size=128)` 对每个 expert 调用 `LinearW4A16.from_float()`。

---

## 文件列表

| 文件 | 功能 |
|------|------|
| `mini_qwen/kernels/moe_router.py` | Top-K router |
| `mini_qwen/kernels/moe_permute.py` | Token 重排 |
| `mini_qwen/kernels/moe_grouped_gemm.py` | Grouped GEMM（oracle + Triton） |
| `mini_qwen/kernels/moe_unpermute.py` | 加权反重排 |
| `mini_qwen/model/layers/moe.py` | Qwen3MoEBlock |
| `mini_qwen/model/qwen3_moe.py` | Qwen3MoEForCausalLM |
| `mini_qwen/model/loader.py` | `load_moe_from_hf()` |
| `tests/test_moe_kernels.py` | 单元测试 + 性能测试 |

---

## Silent Error 防护

| 检查点 | 通过标准 |
|--------|---------|
| permute round-trip | err < 1e-2 |
| expert_offsets 边界 | `offsets[-1] == T*K`，严格相等 |
| grouped_gemm vs oracle | max abs error < 1e-2 |
| E2E vs HF | 首 token **top-1 token 一致**（logits err 仅信息项，见下） |

遇到数值全错但不 crash，检查：
1. `stable=True` 是否漏写
2. `expert_offsets` off-by-one（`bincount` 缺 `minlength`）
3. `expert_weights [E, D, H]` 方向（正确：`X @ W[e].T`，错误：`X @ W[e]`）

---

## 实测数字（RTX 4090）

### 正确性（pytest tests/test_moe_kernels.py）

| 测试 | 结论 |
|------|------|
| test_moe_router | PASS |
| test_moe_permute_offsets | PASS |
| test_moe_permute_unpermute_roundtrip | PASS |
| test_moe_grouped_gemm_vs_oracle | PASS（max err < 1e-2） |
| test_moe_grouped_gemm_empty_expert | PASS |
| test_qwen3_moe_block_bf16 | PASS |
| test_qwen3_moe_block_w4a16 | PASS（量化后 CUDA 设备正确） |

### 性能（test_moe_grouped_gemm_perf，T=8, E=128, K=8, H=3584, D=1536，RTX 4090）

| 实现 | latency | speedup |
|------|---------|---------|
| for-loop oracle（8× torch.mm） | 4.43 ms | 1.00x |
| Triton kernel（动态 grid） | 0.83 ms | **5.32x** |

动态 grid 关键：只启动 8 个 active expert program（而非全量 128 个），
减少无效 kernel launch，是主要加速来源。

### E2E 正确性（Qwen3-30B-A3B GPTQ-Int4 vs HF BF16）

云端 80GB GPU 上对比（HF BF16 ~60GB 与 GPTQ ~16GB 不同时占用：先跑 HF 取
参考 logits 再释放，后加载 GPTQ）。prompt `"Hello, how are you?"`，首 next-token：

| 项目 | 结果 |
|------|------|
| HF top-5 | `[' I', ' Let', ' What', ' Can', ' How']` |
| Our top-5 | `[' I', ' Let', ' Also', ' What', ' And']` |
| **Top-1 一致** | **True**（` I`）→ PASS |
| Top-5 重合 | 3/5 |
| logits max abs err | 2.5（信息项） |
| logits mean abs err | 0.358（信息项） |

**为什么不用 max abs err < 1e-2 做验收**：HF MoE 用 `torch._grouped_mm` 批量算
expert，本实现按 per-expert 加权求和，BF16 累加顺序不同必然产生 ~0.25–2.5 的
logits 差异（即便两边都是 BF16 无量化也有 ~0.94）。叠加 GPTQ-Int4 量化误差后更大。
因此正确性以 **top-1 token 一致** 为准，logits err 仅作信息项。

### W4A16 量化路径说明

朴素 min-max 量化（`load_moe_from_hf(quantize_w4a16=True)`）质量不足以通过 E2E。
实际 W4A16 上线用**预校准的 GPTQ-Int4 checkpoint**（GPTQModel 产出，desc_act=false）。
GPTQ zero-point 用 v1 的 +1 约定：dequant = `(q - (z+1)) * scale`，加载时 `zero_plus_one=True`
（已经验验证：conv B max_err 0.026 < conv A 0.054）。GPTQModel 还量化了 MoE router
（`mlp.gate`），但 router 需 fp16 精度，loader 会把它反量化成普通 weight。

---

## E2E 使用

```python
import torch
from mini_qwen.model.loader import load_moe_from_gptq

# 推荐：加载预校准的 GPTQ-Int4 checkpoint（~16 GB）
model = load_moe_from_gptq(
    "/path/to/Qwen3-30B-A3B-GPTQ-Int4",
    dtype=torch.bfloat16,
    group_size=128,
    zero_plus_one=True,    # GPTQ v1 zero-point +1 约定
    device="cuda",
)
model.eval()
```

朴素 min-max 量化路径（质量较差，仅供 kernel 验证）：

```python
from mini_qwen.model.loader import load_moe_from_hf

model = load_moe_from_hf(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    quantize_w4a16=True,
    group_size=128,
).cuda()
```
