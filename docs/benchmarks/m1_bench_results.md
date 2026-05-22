# M1 Paged Attention Benchmark Results

**GPU**: NVIDIA GeForce RTX 4090 D  
**PyTorch**: 2.8.0+cu128  
**Date**: 2026-05-20  
**Config**: H_q=16, H_kv=8, D=128, block_size=16  
**Timing**: decode warmup=10/reps=100; prefill warmup=5/reps=30

---

## system_bench

baseline = `torch.cat([k_prev, new_k], dim=2)` + `repeat_interleave` + SDPA  
模拟 HF 原生 decode：每 step 重新分配并拷贝全量 KV（O(seq²) memory traffic）

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

---

## kernel_bench

baseline = contiguous KV（一次性分配，不 cat）+ GQA pre-expand + SDPA  
只测 attention 计算本身，排除 KV 拼接开销

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

---

## prefill_bench

baseline 1 (naive) = `sdpa_kernel(MATH)` + GQA pre-expand，O(n²) 显存，无 Flash  
baseline 2 (fa2)   = `sdpa_kernel(FLASH_ATTENTION)` + GQA pre-expand（PyTorch 2.8 dense SDPA 要求同 num_heads）  
ours = `paged_attn_prefill`（Triton tiled causal，**含 KV cache 写入开销**，两个 baseline 不含）

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

---

## 关键结论

### decode（system_bench / kernel_bench）
- **system_bench** speedup 范围：0.6x (batch=1, seq=512) ～ 6.7x (batch=16, seq=2048)
- **kernel_bench** speedup 范围：0.2x (batch=1, seq=512) ～ 2.2x (batch=16, seq=2048)
- system_bench 包含 `torch.cat` 内存分配开销，差距随 seqlen 增大而扩大
- kernel_bench 在小 seq 时 paged 慢（block_table 间接寻址 cache miss），大 seq 时因避免连续大块分配而胜出

### prefill
- **vs naive O(n²)**：0.64x ～ 19.07x
- **vs FA2**：0.13x ～ 0.64x （< 1 表示比 FA2 慢）
- ours 含 KV cache 写入，两个 baseline 不含；两者均 GQA pre-expand（PyTorch 2.8 限制）
- 工程取舍：paged prefill 支持动态 block 分配和 prefix cache，代价是比 FA2 慢
