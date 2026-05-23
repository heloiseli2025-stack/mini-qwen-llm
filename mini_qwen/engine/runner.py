"""Top-level inference orchestration (implemented in M5)."""
from __future__ import annotations

import torch

from mini_qwen.cache.block_manager import BlockManager
from mini_qwen.cache.kv_cache import KVCache
from mini_qwen.engine.scheduler import Scheduler
from mini_qwen.engine.sequence import Sequence


class ModelRunner:
    """Binds the model to KV caches and provides the prefill / decode interface."""

    def __init__(self, model, kv_caches: list[KVCache], block_manager: BlockManager):
        self.model = model
        self.kv_caches = kv_caches
        self.block_manager = block_manager

        # Infer device from the first parameter of embed_tokens
        self.device = next(model.parameters()).device
        # Ensure KV caches are on the same device
        for kv in kv_caches:
            kv.k_cache = kv.k_cache.to(self.device)
            kv.v_cache = kv.v_cache.to(self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers

    def _build_block_table(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build a [B, max_blocks] int32 block table, filling unallocated positions with -1."""
        max_blocks = max(len(s.block_ids) for s in seqs)
        table = torch.full(
            (len(seqs), max_blocks), -1, dtype=torch.int32, device=self.device
        )
        for i, seq in enumerate(seqs):
            ids = seq.block_ids
            table[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        return table

    # ──────────────────────────────────────────────────────────────────────
    # Public interface

    @torch.no_grad()
    def run_prefill(self, seq: Sequence) -> int:
        """Prefill a single sequence, returning the first output token id."""
        input_ids = torch.tensor(
            [seq.prompt_token_ids], dtype=torch.long, device=self.device
        )  # [1, S]
        block_table = self._build_block_table([seq])   # [1, max_blocks]

        logits = self.model.paged_forward_single_prefill(
            input_ids, self.kv_caches, block_table,
        )  # [1, S, vocab]
        return int(logits[0, -1].argmax().item())

    @torch.no_grad()
    def run_decode(self, seqs: list[Sequence]) -> dict[int, int]:
        """Decode one step for a batch of sequences, returning {seq_id: next_token_id}.

        Performs no block allocation (pre-allocated by Scheduler.step()).
        """
        # The decode input for each sequence is its last output token
        input_ids = torch.tensor(
            [seq.output_token_ids[-1] for seq in seqs],
            dtype=torch.long, device=self.device,
        )  # [B]
        # seq_lens_new = prompt + output (current token count; new K/V position = total_len - 1)
        seq_lens_new = torch.tensor(
            [seq.total_len for seq in seqs],
            dtype=torch.int32, device=self.device,
        )  # [B]
        block_table = self._build_block_table(seqs)    # [B, max_blocks]

        logits = self.model.paged_forward_decode(
            input_ids, self.kv_caches, block_table, seq_lens_new,
        )  # [B, vocab]
        return {seq.seq_id: int(logits[i].argmax().item()) for i, seq in enumerate(seqs)}


# ──────────────────────────────────────────────────────────────────────────────
# Top-level generation loop

def generate_batch(
    runner: ModelRunner,
    scheduler: Scheduler,
    prompts: list[list[int]],
    max_new_tokens: int = 256,
    eos_token_id: int = 151645,
) -> dict[int, list[int]]:
    """Continuous batching inference, returning {seq_id: output_token_ids}."""
    for i, p in enumerate(prompts):
        scheduler.add(Sequence(seq_id=i, prompt_token_ids=p))

    outputs: dict[int, list[int]] = {}

    while scheduler.running or scheduler.waiting:
        seqs, mode = scheduler.step()
        if not seqs:
            break   # requests in waiting but KV blocks exhausted; simplified handling: stop

        if mode == "prefill":
            for seq in seqs:
                first_tok = runner.run_prefill(seq)
                seq.output_token_ids.append(first_tok)
                scheduler.promote_to_running(seq)
                # Immediately check EOS / max_new_tokens after prefill completes
                if first_tok == eos_token_id or len(seq.output_token_ids) >= max_new_tokens:
                    outputs[seq.seq_id] = seq.output_token_ids[:]
                    scheduler.finish(seq)
        else:
            next_toks = runner.run_decode(seqs)
            done: list[Sequence] = []
            for seq in seqs:
                tok = next_toks[seq.seq_id]
                seq.output_token_ids.append(tok)
                if tok == eos_token_id or len(seq.output_token_ids) >= max_new_tokens:
                    outputs[seq.seq_id] = seq.output_token_ids[:]
                    done.append(seq)
            for seq in done:
                scheduler.finish(seq)

    return outputs
