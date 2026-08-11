"""
config.py

Central configuration for the LLM-as-Judge pipeline.

All secrets and model choices come from environment variables (see .env.example).
Judge and generator are configured completely independently so that:
  (a) you can point the judge at a different model family than the generator
      (self-enhancement bias mitigation), and
  (b) you can swap models without touching any pipeline code.

Design note: this module intentionally does NOT read the .env file itself.
Call `load_dotenv()` (from python-dotenv) once in main.py before importing
config values, or export the variables in your shell. Keeping config.py
free of file I/O side effects makes it trivial to unit test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Provider = Literal["anthropic", "openai", "gemini"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SUITES_DIR = PROJECT_ROOT / "suites"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_DIR = PROJECT_ROOT / "validation"

for _dir in (SUITES_DIR, LOGS_DIR, REPORTS_DIR, VALIDATION_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value or ""


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an int, got {raw!r}") from exc


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model role (judge OR generator)."""

    provider: Provider
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024
    # Cost per 1M tokens, used only for the cost estimator in report.py.
    # These are rough, user-editable placeholders — real pricing changes
    # often enough that hardcoding "current" numbers would go stale fast.
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError(f"Model name is empty for provider={self.provider}")
        if not self.api_key:
            raise ConfigError(
                f"No API key set for provider={self.provider}. "
                f"Set {self.provider.upper()}_API_KEY in your environment."
            )


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    judge: ModelConfig
    generator: ModelConfig
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    run_both_orders: bool = True  # position-bias mitigation, on by default
    request_timeout_seconds: float = 60.0


_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_judge_config() -> ModelConfig:
    provider = _get_env("JUDGE_PROVIDER", "anthropic", required=True).lower()
    api_key_env = _PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    return ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model=_get_env("JUDGE_MODEL", required=True),
        api_key=_get_env(api_key_env, required=True),
        temperature=_get_env_float("JUDGE_TEMPERATURE", 0.0),
        max_tokens=_get_env_int("JUDGE_MAX_TOKENS", 1024),
        input_cost_per_million=_get_env_float("JUDGE_INPUT_COST_PER_MILLION", 0.0),
        output_cost_per_million=_get_env_float("JUDGE_OUTPUT_COST_PER_MILLION", 0.0),
    )


def load_generator_config() -> ModelConfig:
    provider = _get_env("GENERATOR_PROVIDER", "openai", required=True).lower()
    api_key_env = _PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    return ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model=_get_env("GENERATOR_MODEL", required=True),
        api_key=_get_env(api_key_env, required=True),
        temperature=_get_env_float("GENERATOR_TEMPERATURE", 0.7),
        max_tokens=_get_env_int("GENERATOR_MAX_TOKENS", 1024),
        input_cost_per_million=_get_env_float("GENERATOR_INPUT_COST_PER_MILLION", 0.0),
        output_cost_per_million=_get_env_float("GENERATOR_OUTPUT_COST_PER_MILLION", 0.0),
    )


def load_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        judge=load_judge_config(),
        generator=load_generator_config(),
        max_retries=_get_env_int("MAX_RETRIES", 3),
        retry_backoff_seconds=_get_env_float("RETRY_BACKOFF_SECONDS", 1.5),
        run_both_orders=_get_env("RUN_BOTH_ORDERS", "true").lower() != "false",
        request_timeout_seconds=_get_env_float("REQUEST_TIMEOUT_SECONDS", 60.0),
    )


# Explicit rubric used across the pipeline. Centralized here so judge.py,
# prompts.py, and report.py all agree on criterion names and score bounds.
RUBRIC_CRITERIA: tuple[str, ...] = (
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone_safety",
)
SCORE_MIN = 0
SCORE_MAX = 10
