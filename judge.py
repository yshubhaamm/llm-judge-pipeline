"""
judge.py

The core pairwise LLM-as-judge engine.

Key design choices:
- `LLMClient` is a tiny provider-agnostic wrapper. Judge and generator can
  each use Anthropic or OpenAI independently (self-enhancement mitigation:
  point the judge at a different model family than whatever produced the
  outputs being judged).
- `PairwiseJudge.judge_pair()` runs ONE ordering. `PairwiseJudge.judge_pair_both_orders()`
  runs A-vs-B and B-vs-A and reports whether the verdict flipped
  (position-bias measurement), per the assignment's mandatory bias check.
- Every call is logged (prompt + raw response + parsed verdict + timestamp)
  via `utils.JsonlLogger` before any parsing is attempted, so a malformed
  response is still auditable.
- Retries happen at the parse layer: if parser.parse_verdict() fails, we
  re-send the same prompt (optionally with a stricter reminder appended)
  up to `max_retries` times before giving up on that case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from config import ModelConfig, PipelineConfig
from parser import ParseError, SchemaError, Verdict, parse_verdict
from prompts import build_pairwise_prompt, build_retry_reminder
from utils import JsonlLogger, TokenTracker, utc_timestamp


# ---------------------------------------------------------------------------
# Provider-agnostic LLM client
# ---------------------------------------------------------------------------

class ChatClient(Protocol):
    """Minimal interface every provider adapter must satisfy."""

    def complete(self, system: str, user: str, config: ModelConfig) -> "ChatResult":
        ...


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


class AnthropicClient:
    """Adapter around the Anthropic Messages API."""

    def __init__(self) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install with: pip install anthropic"
            ) from exc
        self._sdk = anthropic

    def complete(self, system: str, user: str, config: ModelConfig) -> ChatResult:
        client = self._sdk.Anthropic(api_key=config.api_key)
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return ChatResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIClient:
    """Adapter around the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install with: pip install openai"
            ) from exc
        self._sdk = openai

    def complete(self, system: str, user: str, config: ModelConfig) -> ChatResult:
        client = self._sdk.OpenAI(api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return ChatResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class GeminiClient:
    """Adapter around Google's Gemini API via the current `google-genai` SDK.

    (The older `google-generativeai` package is deprecated by Google as of
    2026 — this uses its replacement, `google.genai.Client`.) System
    instruction and generation params both go through `GenerateContentConfig`
    on each call; token usage comes back on `response.usage_metadata`.
    """

    def __init__(self) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for GeminiClient. "
                "Install with: pip install google-genai"
            ) from exc
        self._genai = genai
        self._types = types

    

    def complete(self, system: str, user: str, config: ModelConfig) -> ChatResult:
        import time

        client = self._genai.Client(api_key=config.api_key)

        max_attempts = 6

        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=config.model,
                    contents=user,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                      
                        max_output_tokens=max(config.max_tokens, 512),
                        response_mime_type="application/json",
                    ),
                )

                text = response.text or ""
                usage = getattr(response, "usage_metadata", None)

                input_tokens = (
                    getattr(usage, "prompt_token_count", 0) if usage else 0
                )
                output_tokens = (
                    getattr(usage, "candidates_token_count", 0) if usage else 0
                )

                # Small delay to stay under Gemini free-tier rate limit
                time.sleep(6)

                return ChatResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except Exception as e:
                msg = str(e)

                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    wait = 8 * (attempt + 1)
                    print(f"[Gemini] Rate limit hit. Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue

                raise

        raise RuntimeError(
            "Gemini API rate limit persisted after multiple retries."
        )

def make_client(provider: str) -> ChatClient:
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Pairwise judging
# ---------------------------------------------------------------------------

@dataclass
class CaseInput:
    """One test case to be judged."""

    case_id: str
    input_prompt: str
    system_prompt: str
    output_a: str
    output_b: str
    expected_output: str | None = None
    criteria_notes: str | None = None


@dataclass
class PositionBiasResult:
    case_id: str
    verdict_a_first: Verdict
    verdict_b_first: Verdict
    flipped: bool
    final_winner: str  # after normalizing B-first verdict back to A/B labels


class PairwiseJudge:
    """Runs pairwise A-vs-B judging with retry, logging, and position-bias checks."""

    def __init__(self, pipeline_config: PipelineConfig, client: ChatClient | None = None):
        self.config = pipeline_config
        self.client = client or make_client(pipeline_config.judge.provider)
        self.prompt_logger = JsonlLogger("logs/judge_prompts.log")
        self.response_logger = JsonlLogger("logs/judge_responses.log")
        self.token_tracker = TokenTracker()

    def _call_judge_once(self, system: str, user: str) -> ChatResult:
        return self.client.complete(system=system, user=user, config=self.config.judge)

    def _judge_with_retries(self, case: CaseInput, swap: bool) -> Verdict:
        """Send the pairwise prompt, retrying on parse/schema failure.

        `swap` controls whether output_a/output_b are sent in original
        order or swapped, for position-bias measurement.
        """
        a_label, b_label = ("B", "A") if swap else ("A", "B")
        text_a, text_b = (case.output_b, case.output_a) if swap else (case.output_a, case.output_b)

        system, user = build_pairwise_prompt(
            input_prompt=case.input_prompt,
            system_prompt=case.system_prompt,
            output_a=text_a,
            output_b=text_b,
            expected_output=case.expected_output,
            criteria_notes=case.criteria_notes,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            prompt_for_attempt = user if attempt == 1 else user + "\n\n" + build_retry_reminder()

            self.prompt_logger.write(
                {
                    "case_id": case.case_id,
                    "swap": swap,
                    "attempt": attempt,
                    "system": system,
                    "user": prompt_for_attempt,
                    "timestamp": utc_timestamp(),
                }
            )

            result = self._call_judge_once(system, prompt_for_attempt)
            self.token_tracker.record(result.input_tokens, result.output_tokens)

            self.response_logger.write(
                {
                    "case_id": case.case_id,
                    "swap": swap,
                    "attempt": attempt,
                    "raw_response": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "timestamp": utc_timestamp(),
                }
            )

            try:
                verdict = parse_verdict(result.text)
            except (ParseError, SchemaError) as exc:
                last_error = exc
                time.sleep(self.config.retry_backoff_seconds * attempt)
                continue

            # If we swapped the inputs, remap the verdict's A/B labels back
            # to the caller's original A/B so downstream code never has to
            # think about swap order.
            if swap:
                verdict = Verdict(
                    winner={"A": "B", "B": "A", "tie": "tie"}[verdict.winner],
                    scores={"A": verdict.scores["B"], "B": verdict.scores["A"]},
                    rationale=verdict.rationale,
                )
            return verdict

        raise ParseError(
            f"Judge failed to produce a valid verdict for case {case.case_id} "
            f"after {self.config.max_retries} attempts. Last error: {last_error}",
            raw_response="",
        )

    def judge_pair(self, case: CaseInput) -> Verdict:
        """Judge a single case in its original A/B order (no position-bias check)."""
        return self._judge_with_retries(case, swap=False)

    def judge_pair_both_orders(self, case: CaseInput) -> PositionBiasResult:
        """Judge a case twice — A-vs-B and B-vs-A — and report the flip.

        This is the mandatory position-bias measurement: the two verdicts
        are compared after remapping labels back to the original A/B, so
        `flipped=True` means the judge's preference genuinely depended on
        presentation order, not just relabeling.
        """
        verdict_first = self._judge_with_retries(case, swap=False)
        verdict_second = self._judge_with_retries(case, swap=True)

        flipped = verdict_first.winner != verdict_second.winner
        # When flipped, we don't have a principled way to pick a winner
        # from disagreement alone; report both and let report.py decide
        # the aggregation policy (e.g. majority, or mark as "contested").
        final_winner = verdict_first.winner if not flipped else "contested"

        return PositionBiasResult(
            case_id=case.case_id,
            verdict_a_first=verdict_first,
            verdict_b_first=verdict_second,
            flipped=flipped,
            final_winner=final_winner,
        )
