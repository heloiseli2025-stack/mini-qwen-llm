# M5 Continuous Batching Scheduler

## 设计目标

消除传统 static batching 中"最短序列等最长序列完成"的空泡浪费。  
本实现范围：prefill/decode 分离、paged KV cache 动态分配、不支持抢占（preemption）。

---

## 架构概览

```
Scheduler ─┬─ waiting  deque[Sequence]
           └─ running  list[Sequence]
                 ↓
         step() → (seqs, mode)
                 ↓
ModelRunner.run_prefill(seq)   → int          # 单条序列
ModelRunner.run_decode(seqs)   → dict[id,tok] # 批量
                 ↓
generate_batch() 顶层循环
```

---

## 关键设计决策

### 1. Decode 优先（Running First）

`step()` 逻辑：**running 非空时先做 decode，running 为空时才取 waiting 做 prefill。**

理由：
- Prefill 大 batch 消耗大量 KV block，若 running 序列等待 prefill 完成则 GPU 利用率下降。
- Decode 阶段每步只生成 1 token，KV block 增长慢，适合多序列并发。

替代方案（prefill 优先）的缺点：running 序列需要等新 prefill 吃满 KV block 后才能继续 decode，增加尾延迟。

### 2. Block 预分配在 Scheduler.step() 中完成

decode 前，`step()` 检查每条 running 序列是否需要新 page：

```python
if total % block_size == 0:   # 当前 token 恰好填满最后一页
    append_block(seq.seq_id)
    seq.block_ids.append(new_block)
```

`run_decode()` 内部**不做任何分配**，只读 `seq.block_ids`。

理由：
- 分配操作（free list pop）在 CPU 上完成，与 GPU kernel 串行较便宜。
- 放在 step() 里能统一管理 OOM 情况（block 不足时跳过该序列，不在 kernel 内部崩溃）。
- 避免 inference 中途出现 RuntimeError。

### 3. Prefill 串行（单条序列）

当前实现每次只 prefill 一条序列（B=1），原因：

- `paged_attn_prefill` kernel 假设输入为 packed token，需要 `cu_seqlens` 处理变长 batch。
- `fused_qkv_rope` 使用 `seq_pos = pid_tok % S`，只在所有序列等长时正确。
- 多序列并发 prefill 要求 padded batch 或 varlen FlashAttention，属于 M6 工作。

代价：多个 prompt 排队时 prefill 吞吐略低。收益：decode 时最大化并发度（batched decode）。

### 4. Decode RoPE：绕过 fused_qkv_rope

Decode 阶段每条序列 RoPE 位置不同，`fused_qkv_rope` 内部 `seq_pos = pid_tok % S`（S=1 时永远为 0）不适用。

修复：decode branch 使用**unfused 路径**：
1. 独立 QKV projection
2. QK-Norm（per-head RMSNorm）
3. 按各序列位置 index `cos_cached / sin_cached`，得 `[B, head_dim]`，广播到 `[B, 1, 1, head_dim]`
4. 手动 `rotate_half` + 乘加
5. 调 `write_kv_decode` 写入 cache
6. 调 `paged_attn_decode` 读取 cache 计算注意力

### 5. OOM 降级策略

- prefill：`block_manager.num_free_blocks < num_blocks_needed` 时新序列**留在 waiting**，不抢占 running。
- decode：block 不足时跳过（`continue`），该序列继续用旧 block，下一步重试。
- `generate_batch` 在 `seqs` 为空时 break，避免死循环。

---

## 数据流

### Prefill

```
input_ids [1, S]
   └─ embed → [1, S, H]
   └─ rotary_emb(S) → cos/sin [S, D]
   └─ 逐层 paged_forward(..., cu_seqlens=[0,S], mode="prefill")
       └─ fused_qkv_rope → q/k/v [1, S, H_*, D]
       └─ paged_attn_prefill → 写 KV cache + 计算注意力输出 [total, H_q, D]
   └─ norm → lm_head → logits [1, S, vocab]
取 logits[0, -1] 作为首个输出 token
```

### Decode

```
input_ids [B]  (每条序列上一步的 output token)
   └─ embed → [B, 1, H]
   └─ cos_cached[positions] → cos [B, D]  （positions = seq_lens_new - 1）
   └─ 逐层 paged_forward(..., seq_lens_new=[B], mode="decode")
       └─ unfused QKV + per-seq RoPE
       └─ write_kv_decode → 写新 K/V 到 paged cache
       └─ paged_attn_decode → 读全历史 K/V，输出 [B, H_q, D]
   └─ norm → lm_head → logits [B, vocab]
各序列独立 argmax → next token
```

---

## 性能参数（A800 80GB 推荐配置）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_seqs_in_flight` | 64–128 | A800 80GB 可容纳更多并发序列 |
| `max_prefill_tokens` | 4096 | 单次 prefill token 预算 |
| `block_size` | 16 | 冻结（§3.5.1），不得修改 |
| `num_blocks` | 2048+ | 视模型层数、KV head 数调整 |

---

## 验收结果

| 测试 | 结果 |
|------|------|
| `test_scheduler_state_machine` | waiting→running→finished 状态转移正确，block 释放后 free_blocks 恢复 |
| `test_block_oom` | 超容量请求留在 waiting，不崩溃 |
| `test_generate_batch_mock` | 3 条序列 mock runner 全部返回，block 全释放 |
| `test_throughput_vs_sequential` | GPU 上 batched ≥ 5x sequential（待服务器实测填入） |

*实测吞吐数字待服务器运行 `bench_throughput.py` 后更新。*
