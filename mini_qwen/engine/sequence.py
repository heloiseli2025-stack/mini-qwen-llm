"""Sequence state machine.

§3.5.3 Frozen interface — field definitions must not be modified without owner approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class Sequence:
    seq_id: int                                              # immutable
    prompt_token_ids: list[int]                              # immutable
    output_token_ids: list[int] = field(default_factory=list)   # append-only
    block_ids: list[int] = field(default_factory=list)           # append-only
    status: Literal["waiting", "running", "finished"] = "waiting"

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
