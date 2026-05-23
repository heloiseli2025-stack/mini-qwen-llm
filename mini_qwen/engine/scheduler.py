"""Continuous Batching scheduler (implemented in M5)."""
from __future__ import annotations

import math
from collections import deque

from mini_qwen.cache.block_manager import BlockManager
from mini_qwen.engine.sequence import Sequence


class Scheduler:
    """Prefill/Decode separated scheduler.

    Policy: when running is non-empty, prioritize decode; otherwise take a batch from waiting for prefill.
    Block pre-allocation is done inside step(); run_decode performs no allocation.
    Preemption is not supported (new requests stay in waiting on OOM).
    """

    def __init__(
        self,
        block_manager: BlockManager,
        max_seqs_in_flight: int = 16,
        max_prefill_tokens: int = 4096,
    ):
        self.block_manager = block_manager
        self.max_seqs = max_seqs_in_flight
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def step(self) -> tuple[list[Sequence], str]:
        """Return the list of sequences to process this step and the mode ("prefill" or "decode")."""
        if self.running:
            # Before decode, pre-allocate a new block for each sequence (write position = total_len-1; allocate if it's the first slot of a new page)
            for seq in self.running:
                total = seq.total_len  # current token count; new K/V write position = total_len - 1
                if (total - 1) % self.block_manager.block_size == 0:
                    if self.block_manager.num_free_blocks == 0:
                        continue   # OOM: skip, let the sequence continue using the tail of the old block
                    new_block = self.block_manager.append_block(seq.seq_id)
                    seq.block_ids.append(new_block)
            return list(self.running), "decode"

        # prefill: take as many as possible from waiting (bounded by max_seqs and max_prefill_tokens)
        batch: list[Sequence] = []
        total_tokens = 0
        while self.waiting and len(batch) < self.max_seqs:
            seq = self.waiting[0]
            seq_len = len(seq.prompt_token_ids)
            if total_tokens + seq_len > self.max_prefill_tokens and batch:
                break
            num_blocks = math.ceil(seq_len / self.block_manager.block_size)
            if self.block_manager.num_free_blocks < num_blocks:
                break   # OOM: new request stays in waiting
            self.waiting.popleft()
            self.block_manager.allocate(seq.seq_id, seq_len)
            seq.block_ids.extend(self.block_manager.get_block_ids(seq.seq_id))
            total_tokens += seq_len
            batch.append(seq)
        return batch, "prefill"

    def promote_to_running(self, seq: Sequence) -> None:
        seq.status = "running"
        self.running.append(seq)

    def finish(self, seq: Sequence) -> None:
        seq.status = "finished"
        self.running.remove(seq)
        self.block_manager.free(seq.seq_id)
