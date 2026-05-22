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
