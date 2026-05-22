"""Continuous Batching 调度器（M5 阶段实现）。"""
from __future__ import annotations

import math
from collections import deque

from mini_qwen.cache.block_manager import BlockManager
from mini_qwen.engine.sequence import Sequence


class Scheduler:
    """Prefill/Decode 分离调度器。

    策略：running 非空时优先做 decode；否则从 waiting 取一批做 prefill。
    Block 预分配在 step() 内完成，run_decode 内部不做任何分配。
    不支持抢占（OOM 时新请求留在 waiting）。
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
        """返回本步骤要处理的序列列表 + 模式（"prefill" 或 "decode"）。"""
        if self.running:
            # decode 前为每条序列预分配新 block（新 token 写入位置 = total_len-1，若为新页首槽则分配）
            for seq in self.running:
                total = seq.total_len  # 当前已有 token 数；新 K/V 写入位置 = total_len - 1
                if (total - 1) % self.block_manager.block_size == 0:
                    if self.block_manager.num_free_blocks == 0:
                        continue   # OOM：跳过，让该序列使用旧 block 末尾
                    new_block = self.block_manager.append_block(seq.seq_id)
                    seq.block_ids.append(new_block)
            return list(self.running), "decode"

        # prefill：从 waiting 取尽可能多（受 max_seqs 和 max_prefill_tokens 双重限制）
        batch: list[Sequence] = []
        total_tokens = 0
        while self.waiting and len(batch) < self.max_seqs:
            seq = self.waiting[0]
            seq_len = len(seq.prompt_token_ids)
            if total_tokens + seq_len > self.max_prefill_tokens and batch:
                break
            num_blocks = math.ceil(seq_len / self.block_manager.block_size)
            if self.block_manager.num_free_blocks < num_blocks:
                break   # OOM：新请求留在 waiting
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
