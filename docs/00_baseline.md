# Qwen3-0.6B 基线：层形状、参数量与计算量

> M0 文档。数据来源：HF config + 手工计算，实际 benchmark 数字待 M0 跑通后填入。

## 模型配置

| 参数 | 值 |
|------|----|
| vocab_size | 151,936 |
| hidden_size | 1,024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8（GQA，2:1 ratio） |
| head_dim | 128 |
| intermediate_size | 3,072 |
| max_position_embeddings | 40,960 |
| rope_theta | 1,000,000.0 |
| tie_word_embeddings | True |
| rms_norm_eps | 1e-6 |

## 各层 Weight Shape

### Embedding
| 模块 | Shape | 参数量 |
|------|-------|--------|
| model.embed_tokens | [151936, 1024] | 155,581,440 |

### 单个 Decoder Layer（× 28）

| 模块 | Shape | 参数量 |
|------|-------|--------|
| input_layernorm | [1024] | 1,024 |
| self_attn.q_proj | [2048, 1024] | 2,097,152 |
| self_attn.k_proj | [1024, 1024] | 1,048,576 |
| self_attn.v_proj | [1024, 1024] | 1,048,576 |
| self_attn.o_proj | [1024, 2048] | 2,097,152 |
| self_attn.q_norm | [128] | 128 |
| self_attn.k_norm | [128] | 128 |
| post_attention_layernorm | [1024] | 1,024 |
| mlp.gate_proj | [3072, 1024] | 3,145,728 |
| mlp.up_proj | [3072, 1024] | 3,145,728 |
| mlp.down_proj | [1024, 3072] | 3,145,728 |
| **单层合计** | | **~12,585,088** |

### 输出层
| 模块 | Shape | 参数量 |
|------|-------|--------|
| model.norm | [1024] | 1,024 |
| lm_head | [151936, 1024] | 155,581,440（与 embed_tokens 共享） |

## 参数量汇总

| 部分 | 参数量 |
|------|--------|
| Embedding（兼作 lm_head） | 155,581,440 |
| 28 × Decoder Layer | 352,382,464 |
| Final RMSNorm | 1,024 |
| **总计（不含共享权重重复计数）** | **≈507,964,928（≈0.51B）** |

## 每 Token 计算量估算（FLOPs，decode，batch=1，seq=2048）

| 算子 | FLOPs（单层） | 说明 |
|------|--------------|------|
| Q proj | 2 × 1 × 1024 × 1024 = 2.1M | |
| K proj | 2 × 1 × 1024 × 512 = 1.0M | GQA kv_heads=8 |
| V proj | 2 × 1 × 1024 × 512 = 1.0M | |
| Attention（decode） | 2 × 16 × 2048 × 64 = 4.2M | seq_len=2048 |
| O proj | 2 × 1 × 1024 × 1024 = 2.1M | |
| Gate + Up proj | 2 × 2 × 1024 × 3072 = 12.6M | |
| Down proj | 2 × 1 × 3072 × 1024 = 6.3M | |
| **单层合计** | **≈29.3M** | |
| **28 层合计** | **≈820M** | |

## 显存估算（BF16，batch=1，seq=2048）

| 部分 | 大小 |
|------|------|
| 模型权重 BF16 | 508M × 2 bytes ≈ **1.02 GB** |
| KV Cache（28层，seq=2048） | 28 × 2 × 2048 × 8 × 64 × 2 bytes ≈ **117 MB** |
| 激活值（估算） | ~50 MB |
| **总计** | **≈1.2 GB** |

*W4A16 量化后权重约 255 MB，显存压缩至约 400 MB（仅权重）。*

## M0 实测数据（待填）

跑通后用 `scripts/compare_with_hf.py` 填入以下数字：

| 指标 | 数值 |
|------|------|
| max abs error（fp32，"你好"） | TODO |
| 首 token latency（CPU，seq=3） | TODO |
| generate 10 tokens 总耗时（CPU） | TODO |
