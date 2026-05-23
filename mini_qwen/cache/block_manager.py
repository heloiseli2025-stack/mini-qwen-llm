"""Physical Block allocation and deallocation.

§3.5.2 Frozen interface:
    block_table[seq_id, virtual_block_idx] = physical_block_id
    shape: [max_num_seqs, max_blocks_per_seq], dtype=int32
    -1 indicates unallocated.
"""
from __future__ import annotations

import math


class BlockManager:
    """Manages allocation and deallocation of physical KV cache blocks.

    Uses a free-list for O(1) allocation and deallocation.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._seq_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate enough physical blocks to hold num_tokens for seq_id, returning a list of block ids.

        Raises RuntimeError (OOM) if there are insufficient free blocks.
        """
        num_needed = math.ceil(num_tokens / self.block_size)
        if len(self._free) < num_needed:
            raise RuntimeError(
                f"KV cache OOM: need {num_needed} blocks, only {len(self._free)} remaining"
            )
        blocks = [self._free.pop() for _ in range(num_needed)]
        self._seq_blocks.setdefault(seq_id, []).extend(blocks)
        return blocks

    def append_block(self, seq_id: int) -> int:
        """Append one new physical block to an existing seq_id (on-demand expansion during decode)."""
        if not self._free:
            raise RuntimeError("KV cache OOM: no free blocks")
        block_id = self._free.pop()
        self._seq_blocks.setdefault(seq_id, []).append(block_id)
        return block_id

    def free(self, seq_id: int) -> None:
        """Release all physical blocks held by seq_id."""
        self._free.extend(self._seq_blocks.pop(seq_id, []))

    def get_block_ids(self, seq_id: int) -> list[int]:
        """Return the list of physical block ids currently held by seq_id."""
        return self._seq_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)
