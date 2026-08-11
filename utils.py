"""
utils.py

Small shared helpers used across the pipeline: an append-only JSONL logger
(for auditable, replayable judge logs) and a token/cost tracker.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-08-11T14:03:22.104Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class JsonlLogger:
    """Append-only JSON-lines logger.

    One JSON object per line, flushed immediately after every write, so
    logs remain readable and replayable even if the process crashes
    mid-run. Thread-safe for simple concurrent use.
    """

    def __init__(self, relative_path: str):
        self.path: Path = PROJECT_ROOT / relative_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


@dataclass
class TokenTracker:
    """Accumulates token usage and estimates cost across judge calls."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    _input_cost_per_million: float = field(default=0.0, repr=False)
    _output_cost_per_million: float = field(default=0.0, repr=False)

    def set_pricing(self, input_cost_per_million: float, output_cost_per_million: float) -> None:
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += input_tokens
        self.total_completion_tokens += output_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (self.total_prompt_tokens / 1_000_000) * self._input_cost_per_million
        output_cost = (self.total_completion_tokens / 1_000_000) * self._output_cost_per_million
        return round(input_cost + output_cost, 6)

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
