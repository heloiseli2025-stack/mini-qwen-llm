# Mini-QwenLLM：高性能 Qwen3 推理与算子优化引擎

> **一行话定位**：基于 Triton + PyTorch 的轻量级 Qwen3 推理加速引擎，单卡 RTX 4090 上跑通 **Qwen3-8B Dense** 与 **Qwen3-30B-A3B MoE**，通过手写 Triton 算子、PagedAttention、W4A16 量化、MoE 专家路由优化、Continuous Batching，对标 vLLM 的核心能力，目标在 4090 单卡上达到 vLLM 60–80% 的吞吐性能。

---

## 0. 给 Claude Code 的协作约定

**这份文档是你和我（项目负责人）的契约。在开始任何编码前，请先：**

1. 通读全文，特别是 §3「技术架构总览」和 §5「里程碑」
2. 遇到任何架构层级的选择（如：先做 paged attention 还是先做 fused kernel），先在 chat 中和我确认，**不要擅自跳过里程碑**
3. 每个里程碑结束时，必须满足 §5 中定义的「验收标准」才进入下一个
4. 每个 Triton kernel **必须配套写一份与 PyTorch 参考实现的数值正确性测试**（rtol=1e-2, atol=1e-2 for bf16），否则不算完成
5. **§3.5 定义的「冻结接口」严禁擅自修改**——这是 AI 协作中最容易出问题的地方。任何对 KV cache shape、kernel 签名、Sequence 数据结构的修改都必须先在 chat 里得到我的明确批准
6. **Debug 时严格遵守 §4.6 的战术手册**——不要跳过「退到最小复现」就开始猜
7. **Benchmark 严格遵守 §4.7 的协议**——不允许偷偷改 prompt 长度或 warmup 让数字好看
8. 我们的目标读者是**面试官**，所以：每写完一个模块，更新该模块对应的 `docs/` 下的设计文档，记录「为什么这么写、对比基线提升多少、Nsight Systems 截图」；遇到的所有 silent error bug 必须写进 `docs/debugging/`

---

## 0.5 Claude Code 模型使用策略

### 启动配置

```bash
claude --model opusplan
```

**opusplan 会根据 plan 模式自动切换模型，不需要你手动 `/model`**：

| 你的状态 | Claude Code 自动使用 |
|---------|---------------------|
| Plan 模式（Shift+Tab 进入） | **Opus 4.7**（架构 / kernel 设计 / 调试推理） |
| Execution 模式（默认） | **Sonnet 4.6**（写代码 / 测试 / 文档） |

Opus 4.7 在 v2.1.117+ 默认就是 xhigh effort，不用额外开。

### 工作流（每个里程碑都这样走）

1. **Shift+Tab 进 plan 模式**，让 Opus 阅读对应模块的 §4 spec
2. 让 Opus 输出实现 plan：grid 形状、BLOCK_SIZE 取值、stride 计算公式、关键指针偏移
3. **人（你）review plan**，确认无误
4. **退出 plan 模式**，Sonnet 接手写实现 + 测试
5. 遇到「能跑但数值不对」→ 立刻 `/model opus`，纯 Opus 手动调试
6. Bug 修完 → `/model opusplan` 切回，继续干活

### 必须手动切到 Opus 4.7 的场景

下面这些任务**即使在 execution 阶段也要手动切到 `/model opus`**：

- 所有标记了 **🔴 [必用 Opus]** 的模块（见 §4）
- 任何涉及**指针间接寻址**的 Triton 代码（block_table 访问、permute 索引）
- 任何**位运算**密集的代码（int4 packing、bit shift）
- 出现「unit test 过了但 e2e 数值不对」这种诡异 bug 时

### 不要用 Haiku 写 kernel 代码

Haiku 4.5 只用于 file lookup、rename、查 Triton API。它**不能**写 paged attention 或 grouped GEMM 这种逻辑——会写出看起来对、跑起来错的代码。

### 怎么告诉 Claude Code "现在该切了"

Claude Code **不会**根据文档里写的"推荐 Opus"自动切换。每个模块的 spec 里我标的 🔴/🟡/🟢 是给**你**看的提示。你在 chat 里明确说一句即可：

> "M3 的 W4A16 dequant kernel 是必用 Opus 的硬骨头，请提醒我先 `/model opus`。"

或者直接在 Claude Code 的 chat 里：

```
/model opus
请按照 MINI_QWEN_LLM.md §4 M3.2 的 spec 写 W4A16 GEMM kernel
```

---

## 1. 项目背景与选型决策

### 1.1 为什么是 Qwen3，不是 Llama-3

| 维度 | Llama-3-8B | Qwen3-8B | Qwen3-30B-A3B (MoE) |
|------|-----------|----------|---------------------|
| 架构 | dense GQA | dense GQA + **QK-Norm** | **MoE**（128 expert / 8 活跃） |
| 单卡 4090 (24GB) 可行性 | ✅ BF16 紧张，W4A16 充裕 | ✅ BF16 紧张，W4A16 充裕 | ❌ BF16 不行，**✅ W4A16 ~15GB 刚好** |
| 中文社区共鸣 | 中 | **高** | **高** |
| MoE 路由优化机会 | ❌ | ❌ | ✅ **唯一硬核新方向** |
| 长上下文 | 8K | 128K (YaRN) | 128K (YaRN) |

**结论**：Qwen3-8B 作为 Dense Baseline，Qwen3-30B-A3B 作为 MoE 进阶目标——单卡 4090 上同时跑通两条技术栈，是 Llama-3 路线给不了的简历差异化。

### 1.2 Qwen3 架构相比 Qwen2 / Llama 的关键改动（必读）

参考：[Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)

- **移除 QKV bias**（Qwen2 有）
- **新增 QK-Norm**：在 Q 和 K 做 attention 之前各过一遍 RMSNorm，提升训练稳定性 → 推理时这是个**额外的算子**，必须融合进 attention prologue
- **MoE 不再有 shared expert**（与 Qwen2.5-MoE 不同）
- **MoE 使用 global-batch load balancing loss**（推理不涉及，但要理解）
- **fine-grained expert segmentation**：Qwen3-MoE = 128 个 expert，每 token 选 top-8
- **GQA 配置**：Qwen3-8B 是 32 Q heads / 8 KV heads（4:1），Qwen3-30B-A3B 类似比例

---

## 2. 目标硬件与基线

### 2.1 硬件假设

- **目标**：单张 RTX 4090（24GB VRAM，830 GB/s HBM 带宽，Ada Lovelace 架构，SM 8.9，**不支持 FP8 Tensor Core 算力优势**，但支持 FP8 数据格式）
- **本地开发**：CPU 或任意低端 GPU，先把 PyTorch 推理 baseline 跑通
- **云端冲刺**：AutoDL / 蓝耘 / RunPod 的 4090 实例，按小时计费

### 2.2 基线对比对象

| Baseline | 用途 |
|----------|------|
| HuggingFace `transformers` 原生推理 | 「最差基线」——我们必须打爆它 |
| `torch.compile` + SDPA | 「中等基线」——必须打过它 |
| **vLLM 0.x** (官方版) | 「天花板」——目标达到 60–80% 吞吐 |

---

## 3. 技术架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                  Mini-QwenLLM Engine (Python)                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Scheduler (Continuous Batching + Prefill/Decode 分离)     │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ KV Cache Manager (Paged, Block-based)                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Qwen3 Model Runner                                        │ │
│  │  ├─ Embedding                                             │ │
│  │  ├─ Decoder Layer × N                                     │ │
│  │  │   ├─ [Triton] RMSNorm                                  │ │
│  │  │   ├─ [Triton] QKV Proj + QK-Norm + RoPE 融合           │ │
│  │  │   ├─ [Triton] Paged Attention (Prefill/Decode 两套)    │ │
│  │  │   ├─ [Triton] O Proj + Residual                        │ │
│  │  │   ├─ [Triton] RMSNorm                                  │ │
│  │  │   ├─ (Dense) [Triton] SwiGLU MLP + W4A16 GEMM          │ │
│  │  │   └─ (MoE)   [Triton] Top-K Router + Grouped GEMM     │ │
│  │  └─ LM Head                                               │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 五大核心攻关模块

| # | 模块 | 难度 | 简历亮点价值 |
|---|------|------|--------------|
| **M1** | Paged Attention（Prefill + Decode 两套 Triton kernel） | ⭐⭐⭐⭐ | 高 |
| **M2** | Fused QKV + QK-Norm + RoPE Kernel | ⭐⭐⭐ | 高（Qwen3 独有） |
| **M3** | W4A16 GEMM Decode Kernel（AWQ 量化） | ⭐⭐⭐⭐ | 极高 |
| **M4** | MoE Fused Router + Grouped GEMM | ⭐⭐⭐⭐⭐ | **核心差异化** |
| **M5** | Continuous Batching Scheduler | ⭐⭐⭐⭐ | 高（系统能力） |

---

## 3.5 冻结接口契约（IMMUTABLE）

> 这一节定义的数据结构和函数签名一旦写定，**严禁擅自修改**。
> 这是 AI agent 协作中最容易翻车的地方：上游改了 shape，下游跟着改，几天后整个工程 drift 到没法 debug。
> Claude Code 在任何时候提议改这里的字段，**必须先在 chat 里得到 owner 明确批准**。

### 3.5.1 KV Cache 物理布局

```python
@dataclass(frozen=True)
class KVCacheConfig:
    num_blocks: int          # 总物理 block 数（启动时一次性分配）
    block_size: int = 16     # 每个 block 容纳的 tokens 数（固定 16，不要改）
    num_kv_heads: int        # GQA 的 KV head 数
    head_dim: int            # 每个 head 的维度
    dtype: torch.dtype       # bf16

# K cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
# V cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
# 注意 layout：block_size 在 num_kv_heads 之前，保证一个 block 内的 tokens 连续存放
```

### 3.5.2 Block Table

```python
# block_table[seq_id, virtual_block_idx] = physical_block_id
# Shape: [max_num_seqs, max_blocks_per_seq], dtype=int32
# 约定：值为 -1 表示该虚拟 block 尚未分配
```

### 3.5.3 Sequence 状态机

```python
@dataclass
class Sequence:
    seq_id: int                           # 不可变
    prompt_token_ids: list[int]           # 不可变
    output_token_ids: list[int]           # 可变（append-only，不允许删元素）
    block_ids: list[int]                  # 可变（append-only）
    status: Literal["waiting", "running", "finished"]
```

### 3.5.4 Kernel 函数签名

所有 Triton kernel 的入参顺序、dtype、shape 一旦定下来就不再改。kernel **内部**实现可任意优化，但**对外签名是契约**。每个 kernel 的签名见对应 `mini_qwen/kernels/*.py` 文件顶部的 `# === FROZEN SIGNATURE ===` 注释块。

---

## 4. 详细技术规格（每个模块）

### M1. Paged Attention

> **推荐模型**：🔴 **必用 Opus 4.7**（无论 plan 还是 execution）
> 理由：decode kernel 涉及 `block_table[seq_id, virtual_block_id]` 间接寻址，GQA broadcast，online softmax 的 `m_i/l_i` 维护——三层复杂度叠加，Sonnet 容易写出能跑但 silent error 的版本。

**目标**：消除 KV Cache 显存碎片，支持任意请求长度的动态拼 batch。

**关键设计**：

- Block size = 16 tokens（vLLM 默认值，**冻结，不要改**）
- 每个 block 存储 `(block_size, num_kv_heads, head_dim)` 的 K 和 V（见 §3.5.1）
- 维护 `block_table: [num_seqs, max_blocks_per_seq]`（见 §3.5.2）
- **两套 attention kernel**：prefill（处理 prompt）+ decode（每步 1 token）

**关键 Triton 知识点**：

- `tl.load(K_ptr + block_offsets, mask=...)`：通过 block_table 间接寻址
- GQA：`num_q_heads_per_kv_head = num_q_heads // num_kv_heads`，一个 KV head 服务多个 Q head
- Online softmax（Flash 风格）：维护 `m_i` (running max) 和 `l_i` (running sum)

### M1 子任务分解（必须按顺序，每步独立验证）

> 不要试图一次写完整 paged attention。这是整个项目最容易翻车的模块。按下面 6 个子任务推进，**前一个的验收不过就不要开始下一个**。

#### M1.0 - Block Manager + Python Oracle（先不写 Triton）

- Python 实现 `BlockManager`：物理 block 池、`allocate(num_blocks)`、`free(seq_id)`
- 用 `torch.einsum` 写一个 naive paged attention reference（**纯 PyTorch，无 Triton**）
- **验收**：与 HF 的 attention 输出 max abs error < 1e-5（fp32）
- 产出：这个 Python 实现**永久保留**作为后续 Triton kernel 的数值 oracle

#### M1.1 - Triton Naive Decode（单 head, BLOCK_SIZE=1, fp32, 无 online softmax）

- 一个 program 处理一个 (seq, head)
- 直接 load 全部 KV 到 SRAM 算（不分 tile）
- **不优化任何东西，只追数值正确性**
- **验收**：与 M1.0 oracle max abs error < 1e-5（fp32）

#### M1.2 - 加入 GQA Broadcast

- 改为多 head：一个 program 处理一个 (seq, q_head_group)
- 同一个 KV head 服务多个 Q head，在 kernel 内 broadcast
- **验收**：与 oracle max abs error < 1e-5

#### M1.3 - 切换到 Online Softmax（FlashAttention v2 风格）

- 维护 running max `m_i` 和 running sum `l_i`
- 分 tile 累加，处理任意长 context
- **验收**：bf16 下与 oracle max abs error < 1e-2

#### M1.4 - 向量化 + BLOCK_SIZE Tuning

- `BLOCK_SIZE_KV ∈ {16, 32, 64, 128}`，autotune
- `tl.load(..., other=0.0)` 配合 mask
- 调 `num_warps` 和 `num_stages`
- **验收**：decode 性能 ≥ PyTorch SDPA + KV concat 的 3x

#### M1.5 - Prefill Kernel（独立 kernel，复用 BlockManager）

- Query 维度变成 `seq_len`，输入 packed `[total_tokens, num_heads, head_dim]`
- 借鉴 FlashAttention v2 的双层循环（外 Q tiles，内 KV tiles）
- **验收**：prefill 性能 ≥ PyTorch SDPA 的 2x

### M1 整体验收

1. 端到端 Qwen3-8B 推理，attention 数值与 HF 对比 max abs error < 1e-2
2. 性能：decode batch=8 seqlen=2048，比 PyTorch SDPA + KV concat 提速 ≥ 3x
3. 显存：相同显存预算下，最大并发 batch ≥ 2x naive 实现

---

### M2. Fused QKV + QK-Norm + RoPE

> **推荐模型**：🟡 **opusplan 即可**
> Plan 阶段（Opus）：设计 5 个算子怎么塞进一个 kernel 的内存布局
> Execution 阶段（Sonnet）：照着 plan 写实现
> 升级到 Opus 的情况：QK-Norm 数值不稳定 / RoPE 角度算错

**目标**：把 attention 前半段（QKV 投影 → split → QK-Norm → RoPE）融合成一个 kernel，避免反复读写 HBM。

**为什么 Qwen3 这步特别重要**：

普通 Llama 是 `x → QKV → split → RoPE → Attention`，共 4 次 HBM 读写。
Qwen3 是 `x → QKV → split → QK-Norm → RoPE → Attention`，多了一次 RMSNorm，**5 次 HBM 读写**。
融合后只需 1 次读 + 1 次写（每个 Q/K/V）。

**关键 Triton 知识点**：

- 一个 program 处理一个 (token, head_group)
- QKV 投影若用 W4A16，则 fused kernel 内部要做 dequant
- RoPE：`cos[pos], sin[pos]` 提前算好缓存好，kernel 内做 `q_rot = q * cos + rotate_half(q) * sin`
- QK-Norm 是 per-head 的 RMSNorm，`weight` shape = `[head_dim]`

**注意**：V 不需要 norm 和 RoPE，所以 V 路径要单独 store。

**验收标准**：

1. 数值正确性：与 HF 参考实现 max abs error < 1e-2
2. 性能：相比未融合的 PyTorch 实现，kernel 数从 5+ 降到 1，端到端 prefill 速度提升 ≥ 1.5x
3. Nsight Systems 截图证明 kernel launch 数减少

---

### M3. W4A16 GEMM Decode Kernel（AWQ 量化）

> **推荐模型**：🔴 **必用 Opus 4.7**（Step 3.2 GEMM kernel 部分），🟡 opusplan 即可（Step 3.1 量化脚本）
> 理由：int4 packing/unpacking 是大量 bit shift 运算（`(qweight >> (i * 4)) & 0xF`），一个 typo 整个数值全错。AWQ scale 应用的方向（per-channel vs per-group）也容易搞混。

**目标**：所有线性层（QKV proj、O proj、MLP up/gate/down）使用 4-bit 权重 + 16-bit 激活，把权重带宽砍掉 75%。

**两步走**：

**Step 3.1 - 离线量化脚本**：
- 实现 AWQ 算法：通过 activation 的 magnitude 反向缩放 weight，保护 salient channels
- 输出格式：`{layer_name: {qweight: int32 packed, scales: bf16, qzeros: int32 packed}}`
- 校准集：使用 wikitext-2 或 c4 的 128 条 sample，每条 seqlen 2048

**Step 3.2 - Triton W4A16 GEMM Kernel**：
- 输入：activation `[M, K] bf16`，packed weight `[K/8, N] int32`，scales `[K/group_size, N] bf16`
- 在 SRAM 内 dequant：`w_fp = (w_int4.to(bf16) - 8) * scales`
- 用 `tl.dot` 调 Tensor Core 做 bf16 GEMM
- 一个 program 处理 `[BLOCK_M, BLOCK_N]` 的输出 tile

**注意 group_size**：AWQ 通常 group_size=128，意味着每 128 个 K 维度共享一组 scale。

**验收标准**：

1. 量化质量：Qwen3-8B 量化后，wikitext-2 PPL 上升 < 0.3
2. 数值正确性：与 `bitsandbytes` 4-bit 实现误差对齐（业界已知误差范围）
3. 性能：在 decode 场景（M=batch_size 通常 ≤ 16）下，W4A16 GEMM 比 BF16 GEMM 快 ≥ 2x
4. 端到端：Qwen3-8B W4A16 显存占用 ≤ 6GB（权重） + KV cache

---

### M4. MoE Fused Router + Grouped GEMM ⭐ 核心差异化

> **推荐模型**：🔴 **必用 Opus 4.7**（全程，尤其 M4.2 Permute 和 M4.3 Grouped GEMM）
> 理由：这是整个项目最难的模块。`expert_offsets` 前缀和、permuted_hidden 的 stride 计算、每个 program 通过 `expert_id * weight_stride` 找自己的 weight——任何一个 stride 算错都会 silent error。Sonnet 在这里几乎一定会翻车。
> **强烈建议**：开 plan 模式让 Opus 把内存布局画成 ASCII 图，确认后再让它（仍然是 Opus）写代码。

**目标**：让 Qwen3-30B-A3B 在 4090 单卡上跑得动且跑得快。

**MoE 推理为什么慢**：

朴素实现：
```python
for expert_id in range(num_experts):  # 128 个 expert
    token_indices = (selected_experts == expert_id).nonzero()
    if len(token_indices) > 0:
        expert_output = experts[expert_id](hidden_states[token_indices])
```
每个 expert 一次小 GEMM，**128 次 kernel launch**，且每个 GEMM 都是细长矩阵，Tensor Core 利用率极低。

**正确做法（按顺序推进，前一步不验证通过不开始下一步）**：

**M4.0 - Naive For-Loop Baseline（必做，作为正确性 oracle）**：
- Python for 循环遍历 expert，调用每个 expert 的 dense forward
- 性能极慢（~50 tok/s），但**这是后续所有 MoE 优化的数值基准**
- 同时打印每个 expert 接收到的 token 数，观察 load balancing
- **验收**：与 HF `Qwen3MoeForCausalLM` 输出 max abs error < 1e-2，端到端能跑通

**M4.1 - Top-K Router Kernel**：
- Input: `hidden_states [num_tokens, hidden_dim]`
- 计算 gate logits = `hidden @ router_weight`
- 取 top-8，softmax，输出 `topk_ids [num_tokens, 8]` 和 `topk_weights [num_tokens, 8]`
- 全部融合成一个 Triton kernel

**M4.2 - Permute Kernel**（关键）：
- 将 tokens 按 expert_id 重排：`permuted_hidden [num_tokens * 8, hidden_dim]`
- 同时记录 `expert_offsets[i]` = expert i 的 tokens 起始位置（前缀和）
- 这样下游 Grouped GEMM 可以连续访存

**M4.3 - Grouped W4A16 GEMM Kernel**：
- 对所有 expert 的 up_proj/gate_proj 同时做 GEMM
- 一个 program 处理一个 expert 的一个 `[BLOCK_M, BLOCK_N]` tile
- 利用 `expert_offsets` 计算该 program 应该处理哪段 tokens
- **核心 trick**：所有 expert 的 weight 在 HBM 中是连续摆放的，program 通过 `expert_id * weight_stride` 找到自己的 weight

**M4.4 - Unpermute + Reduce Kernel**：
- 把 permuted 输出按 token 加权求和：`out[token] = Σ_k topk_weights[token,k] * expert_outputs[k](token)`
- 一个 program 处理一个 token，循环 8 次累加

**性能预期**：
- 朴素 for 循环：~50–100 token/s（基本不可用）
- 优化后：≥ 300 token/s（单卡 4090 + W4A16）

**验收标准**：

1. 数值正确性：与 HF 参考 max abs error < 1e-2
2. 性能：相比 for-loop 实现提速 ≥ 5x
3. 端到端：Qwen3-30B-A3B W4A16 在 4090 上能跑出 ≥ 200 token/s 的 decode 吞吐

---

### M5. Continuous Batching Scheduler

> **推荐模型**：🟡 **opusplan 即可**
> Plan 阶段（Opus）：设计 Sequence 状态机、Scheduler 主循环、prefill/decode 切换逻辑
> Execution 阶段（Sonnet）：Python 调度代码、状态机实现、KV cache 释放逻辑——这些是「系统代码」而非「kernel 代码」，Sonnet 完全够
> 升级到 Opus 的情况：出现 deadlock / starvation / KV cache 不释放等并发 bug

**目标**：消除 batch 内不同长度请求互相等待的浪费（=「early exit + 动态加新请求」）。

**关键设计**：

- 每个 step 维护一个 `running_queue`，里面是当前活跃的 sequence
- **Prefill / Decode 分离**：
  - 一个 step 内要么处理 prefill（一批新请求），要么处理 decode（所有活跃请求各生成 1 token）
  - **Chunked Prefill**（进阶）：把超长 prompt 切成 chunk，每个 step 处理一个 chunk + 其他 decode 请求
- 请求 `done` 后立即从 running_queue 移除，释放其 KV cache blocks，从 waiting_queue 取新请求加入

**简化范围**（项目阶段）：
- 不做抢占（preemption），KV cache 不够就让新请求等
- 不做 prefix caching
- 不做投机解码

**验收标准**：

1. 端到端：发起 100 个不同长度（128–2048 tokens）的请求，全部成功完成
2. 性能：相比 batch=1 sequential 推理，吞吐提升 ≥ 5x
3. 公平性：max latency / median latency < 3

---

## 4.6 Triton Debug 战术手册

> Triton 开发的 90% 时间在 debug，不在写代码。这一节是给 Claude Code 的标准操作流程——遇到问题**严格按顺序**执行，**不要跳步骤瞎猜**。

### 数值不对（silent error）

按顺序执行，每步都不要跳过：

1. **退到最小复现**：`BLOCK_SIZE=1`, `num_warps=1`, `batch=1`, single head, fp32 accumulation
2. **对照 oracle**：所有 kernel 必须有对应的 PyTorch 纯 Python 实现，跑 oracle 对比每层输出
3. **逐层二分定位**：找出第一个数值偏差 > atol=1e-5 的算子
4. **检查 stride**——这里 90% 的 bug 都在：
   - 打印 input/weight/output 的 `.stride()` 与 kernel 内表达式对比
   - 注意 PyTorch 默认 row-major，Triton 的 `tl.dot` 要求特定 layout
5. **检查 mask**：`tl.load(ptr, mask=..., other=0.0)` 的 `mask` 边界条件和 `other` 默认值
6. **检查 transpose**：matmul `(M,K)@(K,N)` vs `(M,K)@(N,K)`，K dim 对齐了吗

### 出现 NaN

按概率排序检查：

1. **Online softmax 数值稳定性**：
   - `m_i_new = max(m_i, m_curr)` 漏写
   - `l_i = l_i * exp(m_i - m_i_new) + l_curr` 的 scale 修正方向写反
2. **exp overflow**：减 max 后才能 exp
3. **除零**：softmax 分母为 0（mask 全 -inf 的情况）
4. **RMSNorm eps 过小**：bf16 下 `eps=1e-6` 可能不够，改 `1e-5`

### W4A16 / 量化「PPL 爆炸」

这是最隐蔽的一类 bug，专门列出来：

1. **int4 unpack 方向**：`(qweight >> (i * 4)) & 0xF` vs `(qweight >> ((7-i) * 4)) & 0xF`——packing 和 unpacking 方向必须一致，否则前 4 个 weight 和后 4 个对调
2. **zero point 偏移**：AWQ 通常是 `(w_int - 8) * scale`，不是 `w_int * scale - 8 * scale`
3. **scale 的 group 维度**：scale shape 是 `[K/group_size, N]` 还是 `[N, K/group_size]`，broadcast 方向决定一切
4. **必须用 bitsandbytes 4-bit 实现作 oracle**，逐层对比 weight dequant 结果

### 性能不达标

1. `triton.testing.do_bench` 测 wall time（注意 warmup ≥ 20 次）
2. `ncu --section LaunchStats --section Occupancy` 看 occupancy 和 register pressure
3. Nsight Systems 看 kernel launch overhead 占比
4. `TRITON_PRINT_AUTOTUNING=1` 看 autotune 选了哪个 config，是否真的在用 Tensor Core
5. 检查 register spill：寄存器溢出会让 latency 暴涨 3-5x

### 编译错误

1. Triton 类型推导经常失败 → 加显式 `.to(tl.float32)`、`.to(tl.bfloat16)`
2. `constexpr` 参数必须是真正的编译时常量，不能从 tensor.shape 里取
3. `tl.dot` 对 input dtype 有严格要求（fp16/bf16/fp32），不能混

---

## 4.7 Benchmark 协议（红线，不可偷工减料）

> AI agent 倾向于「优化 benchmark 而非 kernel」——偷偷改 prompt 长度、缩小 batch、加大 warmup，让数字好看。
> 这一节**冻结**所有 benchmark 配置，任何修改必须在 chat 里向 owner 提请并得到批准。

### 硬件 & 软件环境（固定）

- GPU: RTX 4090, 24GB VRAM
- CUDA: 12.4+
- PyTorch: 2.4+
- Triton: 3.0+
- 单一进程，跑 benchmark 期间无其他 GPU 负载

### 通用计时配置（固定）

- Warmup: **20 iters**（不计入）
- Measurement: **100 iters**
- Report: **median**（不是 mean，避免离群点污染）
- 每次计时前 `torch.cuda.synchronize()`

### Decode Throughput Benchmark

- Prompt length: **2048**（固定，不允许改）
- Generation length: 256
- Batch size: 测 {1, 8, 16, 32}，每个都要报
- Metric: `tokens/s = (gen_len × batch_size) / median_wall_time`

### Prefill Latency Benchmark

- Prompt length: {128, 512, 2048, 8192}（全部要测）
- Batch size: 1
- Metric: 首 token latency (ms)

### 端到端服务吞吐 Benchmark

- 100 个随机请求，prompt_len ∈ Uniform(128, 2048)，gen_len = 256
- 全部完成后算总 token / total wall time
- 同时记录 P50 / P99 latency

### 报告必须包含

1. 完整命令行（`scripts/bench_*.py --batch ... --seqlen ...`）
2. Git commit hash
3. Nsight Systems 报告（`.nsys-rep` 文件存到 `docs/benchmarks/`）
4. 与 HF / vLLM 同一组配置的对比表
5. 显存峰值 (`torch.cuda.max_memory_allocated()`)

### 🚫 红线（违反 = benchmark 作弊）

- ❌ 报告 batch=32 吞吐，但显存只够 batch=8 →  必须按真实可跑的最大 batch 报告
- ❌ Warmup 少于 20 iters
- ❌ 只测一次取最好的（必须 100 次取 median）
- ❌ 为了让数字好看而改 prompt 长度、改 dtype、关 sampling
- ❌ 与 vLLM 对比时用不同的 prompt / batch / dtype

---

## 5. 里程碑（不可跳过）

| 阶段 | 时间 | 主用模型 | 交付物 | 验收 |
|------|------|---------|--------|------|
| **M0 - 筑基** | 第 1 周（本地） | 🟢 opusplan | PyTorch 纯 Python 跑通 Qwen3-0.6B 推理，理解每层 shape，跑通 Triton 官方 tutorial 前 3 个 | 能在 CPU 上输出 Qwen3-0.6B 的 logits，与 HF 输出 max abs error < 1e-4 |
| **M1 - PagedAttention** | 第 2 周（云端） | 🔴 **Opus 4.7** | Paged Attention 两套 kernel + 单元测试 | §M1 验收标准全部通过 |
| **M2 - Fused QKV/RoPE** | 第 3 周 | 🟡 opusplan | Fused kernel + 替换 baseline 中的 attention prologue | §M2 验收标准全部通过 |
| **M3 - W4A16** | 第 4–5 周 | 🔴 **Opus 4.7**（GEMM kernel） / 🟡 opusplan（量化脚本） | AWQ 量化脚本 + W4A16 GEMM kernel + Qwen3-8B 端到端 W4A16 跑通 | §M3 验收标准全部通过；端到端吞吐 ≥ HF 的 3x |
| **M4 - MoE** | 第 6–7 周 | 🔴 **Opus 4.7**（全程） | Router + Permute + Grouped GEMM + Unpermute，Qwen3-30B-A3B 端到端 W4A16 跑通 | §M4 验收标准全部通过 |
| **M5 - Continuous Batching** | 第 8 周 | 🟡 opusplan | Scheduler + Demo Server | §M5 验收标准全部通过 |
| **M6 - 收尾** | 第 9 周 | 🟢 Sonnet 4.6 主导 | README、blog 文章、Nsight 截图、所有 benchmark 数据、面试话术 | 项目可以直接挂简历 |

**图例**：🔴 必用 Opus 4.7（kernel 逻辑硬骨头） / 🟡 opusplan 即可（plan-Opus, execute-Sonnet）/ 🟢 Sonnet 4.6 足够

**节奏原则**：宁可砍掉 M4 或 M5 也要把 M1–M3 做到「面试官无法挑刺」的程度。深度 > 广度。

---

## 6. 代码结构

```
mini-qwen-llm/
├── README.md                      # 给 GitHub 看的入口
├── MINI_QWEN_LLM.md               # 本文档
├── pyproject.toml                 # uv / pdm 管理依赖
├── docs/
│   ├── 01_paged_attention.md      # 每个模块一份「为什么/怎么做/性能数据」
│   ├── 02_fused_qkv_rope.md
│   ├── 03_w4a16_awq.md
│   ├── 04_moe.md
│   ├── 05_scheduler.md
│   ├── debugging/                  # ★ Silent error 失败案例记录（边做边写）
│   │   ├── _template.md            # 标准 bug 报告模板（见下方）
│   │   ├── m1_<bug_name>.md        # 每个遇到的 silent error 一篇
│   │   ├── m3_<bug_name>.md
│   │   └── ...
│   └── benchmarks/
│       ├── nsight_baseline.png
│       ├── nsight_optimized.png
│       └── throughput_comparison.png
├── mini_qwen/
│   ├── __init__.py
│   ├── config.py                   # Qwen3Config, Qwen3MoEConfig
│   ├── model/
│   │   ├── __init__.py
│   │   ├── qwen3.py                # Qwen3ForCausalLM (dense)
│   │   ├── qwen3_moe.py            # Qwen3MoEForCausalLM
│   │   ├── layers/
│   │   │   ├── rms_norm.py
│   │   │   ├── attention.py        # 调用 paged attention kernel
│   │   │   ├── rope.py             # RoPE 缓存预计算
│   │   │   ├── linear_w4a16.py     # W4A16 Linear 替换 nn.Linear
│   │   │   ├── mlp.py              # SwiGLU MLP
│   │   │   └── moe.py              # Qwen3 MoE block
│   │   └── loader.py               # 从 HF safetensors 加载权重
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── paged_attn_prefill.py
│   │   ├── paged_attn_decode.py
│   │   ├── fused_qkv_rope.py
│   │   ├── rms_norm.py             # 备用，可能 PyTorch 实现就够
│   │   ├── swiglu_mlp.py
│   │   ├── w4a16_gemm.py
│   │   ├── moe_router.py
│   │   ├── moe_permute.py
│   │   ├── moe_grouped_gemm.py
│   │   └── moe_unpermute.py
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── awq.py                  # AWQ 量化算法
│   │   └── packing.py              # int4 packing / unpacking utils
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── kv_cache.py             # KVCache class
│   │   └── block_manager.py        # 物理 block 分配 / 回收
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── scheduler.py            # Continuous batching
│   │   ├── sequence.py             # Sequence 状态机
│   │   └── runner.py               # Top-level forward orchestrator
│   ├── server/
│   │   ├── __init__.py
│   │   └── api.py                  # 简单的 FastAPI server（可选）
│   └── utils/
│       ├── __init__.py
│       ├── profiling.py            # torch.profiler / nsys 包装
│       └── sampling.py             # top-p / top-k / temperature
├── tests/
│   ├── conftest.py
│   ├── test_paged_attention.py
│   ├── test_fused_qkv_rope.py
│   ├── test_w4a16_gemm.py
│   ├── test_moe_kernels.py
│   ├── test_scheduler.py
│   └── test_end_to_end.py
├── benchmarks/
│   ├── bench_attention.py
│   ├── bench_w4a16.py
│   ├── bench_moe.py
│   ├── bench_throughput.py         # token/s vs HF / vLLM
│   └── bench_memory.py
└── scripts/
    ├── download_model.py           # 下载 Qwen3-8B / Qwen3-30B-A3B
    ├── quantize_awq.py             # 离线量化
    ├── run_inference.py            # CLI 推理入口
    └── compare_with_hf.py          # 数值对比脚本
```

---

## 7. 环境与依赖

```toml
# pyproject.toml 核心依赖
[project]
dependencies = [
    "torch>=2.4.0",
    "triton>=3.0.0",
    "transformers>=4.51.0",   # 支持 Qwen3
    "safetensors>=0.4.0",
    "numpy",
    "tqdm",
    "datasets",               # 量化校准数据
    "fastapi",                # 可选，做 server
    "uvicorn",                # 可选
]

[project.optional-dependencies]
bench = [
    "vllm",                   # 对比基线
    "matplotlib",
    "pandas",
]
dev = [
    "pytest",
    "pytest-benchmark",
    "ruff",
    "ipython",
]
```

**Python 版本**：3.10 或 3.11（3.12 上 triton 兼容性偶有问题，慎用）

**Docker 镜像建议**：`nvcr.io/nvidia/pytorch:24.10-py3` 或同等版本

---

## 8. 给 Claude Code 的开发规范

### 8.1 编码风格

- 严格 type hints
- 所有 Triton kernel 必须有 docstring 写明：
  - Input shape & dtype
  - Output shape & dtype
  - BLOCK_SIZE 等 meta-param 的取值范围
  - 并行 grid 形状
- 复杂指针运算要写注释，**特别是涉及 strides 的部分**

### 8.2 测试要求

- 每个 kernel 必须有对应的 PyTorch 参考实现作为 oracle
- 测试覆盖小 shape（CI 友好）和大 shape（接近实际推理 shape）
- 数值容差：bf16 用 `rtol=1e-2, atol=1e-2`，fp32 用 `rtol=1e-5, atol=1e-5`

### 8.3 性能数据规范

每个 kernel 完成后，benchmark 输出必须包含：
- Latency（μs）
- Throughput（TFLOPS or GB/s，看是 compute-bound 还是 memory-bound）
- 与 baseline 的对比倍率
- Nsight Systems 报告（生成 `.nsys-rep` 文件，保存到 `docs/benchmarks/`）

### 8.4 Git 规范

- 每完成一个 kernel 一个 PR / commit
- Commit message 用约定式：`feat(paged-attn): implement decode kernel`、`perf(w4a16): improve TFLOPS by 15%`
- 主分支保持可运行

### 8.5 模型权重处理

- **不要把模型权重 commit 进 repo**
- 用 `scripts/download_model.py` 从 HuggingFace 或 ModelScope（国内）下载
- 量化后的权重保存到 `weights/` 目录（.gitignore 之）

### 8.6 Silent Error 失败案例记录（重要）

> 真正做过 Triton 的人都知道：90% 的时间在 debug silent error，不是写代码。
> **每个遇到的、花了 > 30 分钟才定位的 silent error**，都必须在 `docs/debugging/` 写一份案例记录。
> 这不仅帮项目可维护，也是面试时讲「你踩过什么坑」最硬的素材。

#### `docs/debugging/_template.md` 模板

```markdown
# Bug: <一句话症状>

**Module**: M1.3 / M3.2 / M4.2 ...
**Severity**: silent error / NaN / perf regression / OOM
**Time to debug**: 例如 2 hours

## 症状（Symptom）
一段可复现的描述。例如：
- W4A16 Qwen3-8B 量化后 wikitext PPL 从 9.2 飙到 4500+
- 单元测试都过，端到端 generate 出来全是乱码

## 复现步骤（Repro）
```bash
python scripts/run_inference.py --model qwen3-8b-w4a16 --prompt "你好"
# 预期: 正常回复
# 实际: !@#$%^&*()...
```

## 根因（Root Cause）
具体到代码行。例如：
`mini_qwen/quantization/packing.py:42`，int4 unpack 时方向写反：
```python
# 错误版
for i in range(8):
    unpacked[..., i] = (qweight >> (i * 4)) & 0xF
# 正确版（packing 时 high bit 在前，unpack 也要 high bit 在前）
for i in range(8):
    unpacked[..., i] = (qweight >> ((7 - i) * 4)) & 0xF
```

## 修复（Fix）
具体 commit hash。

## 教训（Lesson）
1. int4 packing 和 unpacking **必须用同一个工具函数**，不要分别实现
2. 加 unit test：随机生成 int4，pack → unpack，验证 round-trip 完全一致
```

---

## 9. 关键性能 KPI（简历用）

完成后必须能填入以下数字（举例，实际待 benchmark 确认）：

| 指标 | Baseline (HF) | Ours | 提升 |
|------|---------------|------|------|
| Qwen3-8B BF16 decode throughput (bs=1, seqlen=2048) | ~30 tok/s | ≥ 90 tok/s | 3x |
| Qwen3-8B W4A16 decode throughput (bs=1) | N/A | ≥ 150 tok/s | — |
| Qwen3-8B W4A16 显存占用 | — | ≤ 6GB (weights) | — |
| Qwen3-30B-A3B W4A16 decode throughput | OOM | ≥ 200 tok/s | ∞ |
| Qwen3-8B W4A16 端到端吞吐 (continuous batching, 16 并发) | ~80 tok/s 总 | ≥ 800 tok/s 总 | 10x |
| 相对 vLLM 0.x 的吞吐 | — | ≥ 60% | — |

---

## 10. 面试话术（写给项目负责人）

### 10.1 一句话介绍

> 「我在单卡 4090 上从零实现了一个 Qwen3 推理引擎，对标 vLLM 的核心能力。主要写了 5 个 Triton kernel：Paged Attention、Fused QKV/QK-Norm/RoPE、W4A16 GEMM、MoE Grouped GEMM，以及一个 Continuous Batching 调度器。Qwen3-8B W4A16 端到端 decode 比 HF 快 5 倍，单卡能跑通 Qwen3-30B-A3B MoE 的 W4A16 推理。」

### 10.2 必须能在白板上讲清楚的细节

1. **PagedAttention 中 block_table 的内存布局**，以及 decode kernel 中如何通过 `block_table[seq_id, virtual_block_id]` 间接寻址
2. **W4A16 GEMM 中 4-bit 权重是怎么 packing 的**（8 个 int4 打包进一个 int32），以及为什么在 SRAM 内 dequant 而不是预先全部 dequant
3. **AWQ 为什么有效**（salient channel + activation-aware scaling），以及和 GPTQ 的本质区别
4. **MoE 中 permute 的意义**：把同一个 expert 的 tokens 物理上聚到一起，让 Grouped GEMM 能用 Tensor Core
5. **GQA 中一个 KV head 服务多个 Q head 的实现**：Triton kernel 内如何 broadcast K/V
6. **RoPE 为什么 rotate_half 而不是 sin/cos 直接乘**：实数实现复数乘法
7. **Online softmax 的数值稳定性**：m_i 维护 + scale 修正

### 10.3 可能的追问与准备

| 面试官追问 | 你的回答方向 |
|-----------|------------|
| 你这个跟 vLLM 比差多少？为什么不直接用 vLLM？ | 单卡 4090 上能跑到 vLLM 的 60–80%；项目目的是学习内部机制，不是替代 |
| Bank Conflict 怎么避免的？ | 解释 Triton 中 `tl.dot` 的 swizzle，shared memory 的 padding 策略 |
| 为什么选 AWQ 不选 GPTQ？ | AWQ 推理更快（不需要 reorder），量化质量在 4-bit 略优于 GPTQ |
| 你的 paged attention 有没有 prefix caching？ | 没做，但能说清楚怎么扩展（block_table 哈希索引） |
| MoE 的 load balancing 怎么处理？ | 推理时不处理，是训练侧的事；但能说出 global-batch loss 的设计 |

---

## 11. 风险与降级方案

| 风险 | 降级方案 |
|------|----------|
| Triton 在某个 BLOCK_SIZE 编译失败 | 换 BLOCK_SIZE，或拆分成两个 kernel |
| W4A16 数值精度不达标 | 退回 W8A16（int8 量化），仍比 BF16 快且省 |
| MoE Grouped GEMM 太复杂写不出来 | 退回 expert-by-expert 串行 + CUDA Graph，仍能跑通 Qwen3-30B-A3B |
| Continuous Batching 调度有 bug | 退回 static batching（每个 batch 必须等所有请求都 done） |
| 4090 显存不够 | 优先 Qwen3-8B；Qwen3-30B-A3B 不行的话改用 Qwen3-14B |

**底线**：哪怕只完成 M1 + M2 + M3，做到极致深度，也是一个可以挂简历、能讲 30 分钟的好项目。

---

## 12. 立即开始的第一步（M0 任务清单）

请 Claude Code 按顺序执行：

1. ✅ 创建 §6 的目录骨架（空文件 + `__init__.py`）
2. ✅ 写 `pyproject.toml`，按 §7 配置依赖
3. ✅ 在 `mini_qwen/model/qwen3.py` 中，**用纯 PyTorch + HuggingFace transformers 的 layer 拼出一个 Qwen3 dense 推理流程**，目标：
   - 加载 `Qwen/Qwen3-0.6B` 权重
   - 输入 "你好" 能输出合理 logits
   - 与 `AutoModelForCausalLM.from_pretrained(...)` 的输出 max abs error < 1e-4
4. ✅ 在 `tests/test_end_to_end.py` 中写一个最小测试，跑通上面这个流程
5. ✅ 在本地（CPU 即可）跑通这个测试
6. ✅ 在 `docs/00_baseline.md` 中记录 Qwen3-0.6B 每一层的 shape、参数量、计算量（FLOPs）

完成 M0 后，向我汇报，我们一起确认进入 M1。

---

**最后**：这个项目的价值不在于跑得多快，**在于每一行 Triton 代码你都能在面试官面前讲清楚为什么这么写**。Stay deep, stay honest.
