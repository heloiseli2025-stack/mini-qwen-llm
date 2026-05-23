"""KV Cache physical storage.

§3.5.1 Frozen interface — fields of KVCacheConfig must not be modified without approval.
layout: Both K/V caches have shape [num_blocks, block_size, num_kv_heads, head_dim]
Tokens within a block are stored contiguously (ensures decode kernel memory access efficiency).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KVCacheConfig:
    num_blocks: int
    block_size: int = 16        # Fixed at 16, do not change (§3.5.1 frozen)
    num_kv_heads: int = 8
    head_dim: int = 128   # Empirical value for Qwen3-0.6B/8B (corrected after M0 validation)
    dtype: torch.dtype = torch.bfloat16


class KVCache:
    """Physical KV Cache for a single layer. PagedAttention read/write logic implemented in M1."""

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
