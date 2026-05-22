"""KV Cache 物理存储。

§3.5.1 冻结接口——KVCacheConfig 字段严禁擅自修改。
layout: K/V cache 均为 [num_blocks, block_size, num_kv_heads, head_dim]
block_size 内的 tokens 连续存放（保证 decode kernel 访存效率）。
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KVCacheConfig:
    num_blocks: int
    block_size: int = 16        # 固定 16，不要改（§3.5.1 冻结）
    num_kv_heads: int = 8
    head_dim: int = 128   # Qwen3-0.6B/8B 实测值（M0 验证后修正）
    dtype: torch.dtype = torch.bfloat16


class KVCache:
    """单层的物理 KV Cache。M1 阶段实现 PagedAttention 写入/读取逻辑。"""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        # K cache: [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = torch.zeros(
            config.num_blocks, config.block_size,
            config.num_kv_heads, config.head_dim,
            dtype=config.dtype,
        )
        self.v_cache = torch.zeros(
            config.num_blocks, config.block_size,
            config.num_kv_heads, config.head_dim,
            dtype=config.dtype,
        )
