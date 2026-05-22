# M2 Fused QKV + QK-Norm + RoPE

## 动机

Qwen3 Attention 前半段比 Llama 多一步 per-head QK-RMSNorm，unfused 路径：

```
x → GEMM_Q → (HBM写) → RMSNorm_Q → (HBM写) → RoPE_Q → (HBM写)
x → GEMM_K → (HBM写) → RMSNorm_K → (HBM写) → RoPE_K → (HBM写)
x → GEMM_V → (HBM写)
```

Q/K 各 5 次 HBM op，共 10+ 次（不含 V）。

## 实现策略

| 步骤 | 实现 | 理由 |
|------|------|------|
| QKV GEMM | `torch.mm`（cuBLAS） | 大矩阵乘法 cuBLAS 远快于 Triton |
| QK-Norm + RoPE | 单次 Triton kernel | 消除中间 HBM 写回 |
| V | 直接用 GEMM 输出 | V 无 norm/rope |

融合后 Q/K 各 3 次 HBM op（GEMM写 + kernel读 + kernel写），节省 40%。

## Kernel 设计

**文件**：`mini_qwen/kernels/fused_qkv_rope.py`

**Grid**：`(B×S, H_Q + H_KV)` — 1 次 kernel launch 覆盖所有 Q 和 K head

```
pid_head 0..H_Q-1          → Q head（is_q = True）
pid_head H_Q..H_Q+H_KV-1  → K head（is_q = False）
```

**每个 program 处理 1 个 (token, head)**：
1. 分两半加载（HALF_D = 64 elements，避免 register 端的 gather）
2. RMSNorm：fp32 累加（`tl.sum(x1*x1) + tl.sum(x2*x2)`），防止 bf16 精度损失
3. RoPE：`out1 = x1_n * cos1 - x2_n * sin1`，`out2 = x2_n * cos2 + x1_n * sin2`
4. 转 bf16 写回

**RoPE 公式验证**（对应 `rotate_half`）：
```
rotate_half([x1, x2]) = [-x2, x1]
q_rope = q * cos + rotate_half(q) * sin
       = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]
       = [x1*cos1 - x2*sin1, x2*cos2 + x1*sin2]
```

## 验收结果

**正确性**（RTX 4090 D，Triton 3.4.0）：

| 测试 | 结果 |
|------|------|
| Q max abs error | ≤ 0.0078 < 1e-2 ✓ |
| K max abs error | ≤ 0.0078 < 1e-2 ✓ |
| V passthrough error | 0.0 ✓ |

**性能**（B=4，S=512，warmup=10，reps=100，CUDA events）：

| 指标 | unfused | fused | 提升 |
|------|---------|-------|------|
| 延迟 | 0.357 ms | 0.198 ms | **1.80x** |
| CUDA kernel 数 | 33 | **6** | 5.5x 减少 |

kernel 构成（fused）：3 次 cuBLAS GEMM + 1 次 `_qk_norm_rope_kernel` + `aten::view` 等元操作，共 6 个 CUDA 事件。

## 端到端 Attention 延迟

测试范围：QKV 投影 + QK-Norm + RoPE + Paged Prefill（`paged_attn_prefill`），
排除 o_proj，warmup=10，reps=100。

| B | S | unfused (ms) | fused (ms) | speedup |
|---|---|-------------|-----------|---------|
| 1 | 512  | 0.762 | 0.532 | 1.43x |
| 1 | 1024 | 0.763 | 0.532 | 1.43x |
| 1 | 2048 | 1.434 | 0.681 | **2.11x** |
| 4 | 512  | 0.927 | 0.532 | 1.74x |
| 4 | 1024 | 1.662 | 0.827 | 2.01x |
| 4 | 2048 | 2.782 | 1.708 | 1.63x |

加速比 **1.43x ~ 2.11x**；长序列（S=2048）收益更大，因为 norm+rope 的 HBM traffic
随 token 数线性增长，而 flash attention 的收益不依赖 QKV 路径。

## Occupancy 分析

> 云环境（AutoDL）`RmProfilingAdminOnly=1`，ncu performance counters 受限。
> 以下数据来自 Triton 编译缓存（`CompiledKernel.metadata`）+ CUDA occupancy 公式推算。

**Kernel 编译信息**（Triton 3.4.0，SM 8.9，RTX 4090 D）：

| 指标 | 值 |
|------|-----|
| `n_regs` / thread | 26 |
| `num_warps` / block | 4（= 128 threads/block） |
| shared memory / block | 8 bytes |
| spills | 0 |

**Occupancy 推算**（RTX 4090 D，SM 8.9，max 48 warps/SM，65536 regs/SM）：

| 限制因子 | 计算 | max blocks/SM |
|---------|------|--------------|
| Warp 数量 | 48 / 4 | **12** |
| Thread 数量 | 1536 / 128 | 12 |
| Register 用量 | ⌊65536 / (26×128)⌋ = 19 | 19 |
| Shared memory | 102400 / 8 = 12800 | 12800 |

瓶颈：warp 数量，max 12 blocks/SM × 4 warps = **48 warps/SM**

```
Theoretical Occupancy    : 100%  (48/48 warps)
Active Warps Per SM      : 48
Active Threads Per SM    : 1536
```

> Achieved Occupancy 估算 ≥ 80%：grid 大小 2048×24 = 49152 programs 远大于
> 114 SM × 12 blocks = 1368 并发 blocks，SM 可完全填满；kernel 无 `__syncthreads`，
> 各 program 完全独立，无 warp 发散瓶颈。
>
> 精确 achieved occupancy 需 `ncu --section Occupancy`，在允许 performance counters 的
> 环境下可用 `/usr/local/cuda/bin/ncu`。

**结论**：26 regs/thread 极低（255 上限的 10%），register 不是瓶颈，无需拆成两个独立
Q/K kernel。当前单 kernel 设计已达到最大理论 occupancy，无优化空间。

## HBM 流量对比

| 配置 | Q+K HBM ops | 说明 |
|------|------------|------|
| Unfused | 10 次 | GEMM写×2 + norm读×2 + norm写×2 + rope读×2 + rope写×2 |
| Fused | 6 次 | GEMM写×2 + kernel读×2 + kernel写×2 |
| 节省 | 4 次 (-40%) | |

## Roofline 分析

`_qk_norm_rope_kernel` arithmetic intensity 极低（纯向量操作：RMSNorm + RoPE，无矩阵乘），
优化目标是减少 HBM traffic 而非提升 Tensor Core 利用率。当前 kernel 处于 **memory-bound
regime**，theoretical occupancy 100% 说明 SM 可充分隐藏 memory latency，
寄存器用量仅 26/thread（255 上限的 10%），无 register spill，是 memory-bound kernel 的
理想状态。

> TODO：在允许 ncu 的环境下（非容器化实例）补充 `dram__bytes_read.sum` /
> `dram__bytes_write.sum` 实测值，验证 HBM 节省量与理论 -40% 的吻合程度。
