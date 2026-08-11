"""
parser.py

Turns a raw LLM judge response (a string that is *supposed* to be JSON but
often isn't quite) into a validated verdict dict.

Layered strategy, cheapest first:
  1. Strip markdown code fences (```json ... ``` or ``` ... ```).
  2. Try json.loads on the cleaned string directly.
  3. Regex-extract the first {...} block and try again.
  4. Apply light-touch repairs (trailing commas, single quotes, unquoted
     keys) and try again.
  5. Give up and raise ParseError — the caller (judge.py) is responsible
     for retrying the whole judge call, not for guessing at broken JSON
     forever.

This module never invents field values. If required fields are missing
after successful JSON parsing, that's a schema error, not a parse error,
and is reported separately so logs distinguish "model returned garbage"
from "model returned valid JSON with the wrong shape."
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config import RUBRIC_CRITERIA, SCORE_MAX, SCORE_MIN

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


class ParseError(Exception):
    """Raised when a raw judge response cannot be turned into a verdict dict."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class SchemaError(Exception):
    """Raised when parsed JSON is valid but doesn't match the verdict schema."""

    def __init__(self, message: str, parsed: Any):
        super().__init__(message)
        self.parsed = parsed


@dataclass
class Verdict:
    """Validated, structured output of a single pairwise judging call."""

    winner: str  # "A", "B", or "tie"
    scores: dict[str, dict[str, float]]  # {"A": {criterion: score, ...}, "B": {...}}
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "scores": self.scores,
            "rationale": self.rationale,
        }


def strip_code_fences(text: str) -> str:
    """Remove ```json / ``` fences that models love to wrap output in."""
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def extract_json_block(text: str) -> str:
    """Pull the first {...} span out of a string that may have prose around it."""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ParseError("No JSON object found in response.", raw_response=text)
    return match.group(0)


def repair_common_json_errors(text: str) -> str:
    """Best-effort fixes for the JSON mistakes models actually make.

    Handles: trailing commas before } or ], and unquoted object keys.
    Does NOT attempt to fix mismatched brackets or truncated output —
    those are unrecoverable and should trigger a retry instead.
    """
    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    repaired = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', repaired)
    return repaired


def _try_json_loads(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    return result


def parse_raw_response(raw_response: str) -> dict[str, Any]:
    """Run the full fallback chain and return a raw (unvalidated) dict.

    Raises ParseError if every strategy fails.
    """
    if not raw_response or not raw_response.strip():
        raise ParseError("Empty response from judge model.", raw_response=raw_response)

    cleaned = strip_code_fences(raw_response)

    # Strategy 1: direct parse
    result = _try_json_loads(cleaned)
    if result is not None:
        return result

    # Strategy 2: extract {...} block, then parse
    try:
        block = extract_json_block(cleaned)
    except ParseError:
        block = None
    if block is not None:
        result = _try_json_loads(block)
        if result is not None:
            return result

        # Strategy 3: repair common errors, then parse
        repaired = repair_common_json_errors(block)
        result = _try_json_loads(repaired)
        if result is not None:
            return result

    raise ParseError(
        "All parsing strategies exhausted; response is not recoverable JSON.",
        raw_response=raw_response,
    )


def validate_verdict_schema(parsed: dict[str, Any]) -> Verdict:
    """Validate a parsed dict against the expected verdict schema.

    Expected shape:
        {
          "winner": "A" | "B" | "tie",
          "scores": {
            "A": {"correctness": 0-10, ...all RUBRIC_CRITERIA},
            "B": {"correctness": 0-10, ...all RUBRIC_CRITERIA}
          },
          "rationale": "..."
        }
    """
    missing_top = {"winner", "scores", "rationale"} - parsed.keys()
    if missing_top:
        raise SchemaError(f"Missing top-level keys: {sorted(missing_top)}", parsed)

    winner = str(parsed["winner"]).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise SchemaError(f"winner must be 'A', 'B', or 'tie', got {parsed['winner']!r}", parsed)
    winner = "tie" if winner == "TIE" else winner

    scores = parsed["scores"]
    if not isinstance(scores, dict) or set(scores.keys()) != {"A", "B"}:
        raise SchemaError("scores must have exactly keys 'A' and 'B'", parsed)

    normalized_scores: dict[str, dict[str, float]] = {}
    for side in ("A", "B"):
        side_scores = scores[side]
        missing_criteria = set(RUBRIC_CRITERIA) - set(side_scores.keys())
        if missing_criteria:
            raise SchemaError(
                f"scores['{side}'] missing criteria: {sorted(missing_criteria)}", parsed
            )
        normalized: dict[str, float] = {}
        for criterion in RUBRIC_CRITERIA:
            value = side_scores[criterion]
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise SchemaError(
                    f"scores['{side}']['{criterion}'] is not numeric: {value!r}", parsed
                ) from exc
            if not (SCORE_MIN <= numeric <= SCORE_MAX):
                raise SchemaError(
                    f"scores['{side}']['{criterion}']={numeric} out of range "
                    f"[{SCORE_MIN}, {SCORE_MAX}]",
                    parsed,
                )
            normalized[criterion] = numeric
        normalized_scores[side] = normalized

    rationale = str(parsed["rationale"]).strip()
    if not rationale:
        raise SchemaError("rationale is empty", parsed)

    return Verdict(winner=winner, scores=normalized_scores, rationale=rationale)


def parse_verdict(raw_response: str) -> Verdict:
    """Full pipeline: raw model text -> validated Verdict.

    Raises ParseError (unrecoverable JSON) or SchemaError (valid JSON,
    wrong shape). Callers should treat both as "this attempt failed,
    retry the judge call" — see judge.py's retry loop.
    """
    parsed = parse_raw_response(raw_response)
    return validate_verdict_schema(parsed)
