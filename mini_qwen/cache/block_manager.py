"""物理 Block 分配与回收。

§3.5.2 冻结接口：
    block_table[seq_id, virtual_block_idx] = physical_block_id
    shape: [max_num_seqs, max_blocks_per_seq], dtype=int32
    -1 表示未分配。
"""
from __future__ import annotations

import math


class BlockManager:
    """管理物理 KV cache block 的分配与回收。

    使用 free-list 实现 O(1) 分配和释放。
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._seq_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """为 seq_id 分配足够容纳 num_tokens 的物理 block，返回 block id 列表。

        若空闲 block 不足，抛出 RuntimeError（OOM）。
        """
        num_needed = math.ceil(num_tokens / self.block_size)
        if len(self._free) < num_needed:
            raise RuntimeError(
                f"KV cache OOM：需要 {num_needed} blocks，仅剩 {len(self._free)}"
            )
        blocks = [self._free.pop() for _ in range(num_needed)]
        self._seq_blocks.setdefault(seq_id, []).extend(blocks)
        return blocks

    def append_block(self, seq_id: int) -> int:
        """为已有 seq_id 追加一个新物理 block（decode 阶段按需扩展）。"""
        if not self._free:
            raise RuntimeError("KV cache OOM：无空闲 block")
        block_id = self._free.pop()
        self._seq_blocks.setdefault(seq_id, []).append(block_id)
        return block_id

    def free(self, seq_id: int) -> None:
        """释放 seq_id 占用的全部物理 block。"""
        self._free.extend(self._seq_blocks.pop(seq_id, []))

    def get_block_ids(self, seq_id: int) -> list[int]:
        """返回 seq_id 当前持有的物理 block id 列表。"""
        return self._seq_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)
