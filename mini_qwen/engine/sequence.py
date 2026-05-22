"""Sequence 状态机。

§3.5.3 冻结接口——字段定义严禁擅自修改，改动须经 owner 批准。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class Sequence:
    seq_id: int                                              # 不可变
    prompt_token_ids: list[int]                              # 不可变
    output_token_ids: list[int] = field(default_factory=list)   # append-only
    block_ids: list[int] = field(default_factory=list)           # append-only
    status: Literal["waiting", "running", "finished"] = "waiting"

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
