"""
report.py

Turns raw per-case judging output (Verdicts / PositionBiasResults) into the
three deliverable reports:
  - suite_report.json       : pass rate, mean scores, win rate for one config
  - ab_report.json           : config A vs config B comparison + declared winner
  - position_bias_report.json: flip count / flip rate across a suite

Nothing in this module calls the judge or does I/O beyond writing the final
JSON files — it's pure aggregation over data it's handed, which makes it
straightforward to unit test with synthetic Verdicts.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import REPORTS_DIR, RUBRIC_CRITERIA
from judge import PositionBiasResult
from parser import Verdict


# ---------------------------------------------------------------------------
# Suite report (single config)
# ---------------------------------------------------------------------------

def _mean_scores(verdicts: list[Verdict], side: str) -> dict[str, float]:
    """Mean per-criterion score for one side ('A' or 'B') across verdicts."""
    if not verdicts:
        return {c: 0.0 for c in RUBRIC_CRITERIA}
    return {
        criterion: round(
            statistics.mean(v.scores[side][criterion] for v in verdicts), 3
        )
        for criterion in RUBRIC_CRITERIA
    }


def build_suite_report(
    config_label: str,
    verdicts: dict[str, Verdict],
    pass_threshold: float = 6.0,
) -> dict[str, Any]:
    """Aggregate a suite's verdicts (candidate = side 'A' by convention).

    `verdicts` maps case_id -> Verdict, where side "A" is the candidate
    being evaluated and side "B" is its comparison baseline for this run.
    A case "passes" if the candidate's mean rubric score for that case is
    >= pass_threshold. This mirrors how pass/fail gates are typically
    defined for a single-config suite run.
    """
    all_verdicts = list(verdicts.values())
    n = len(all_verdicts)

    per_case: list[dict[str, Any]] = []
    wins = 0
    passes = 0
    for case_id, v in verdicts.items():
        candidate_mean = statistics.mean(v.scores["A"].values())
        passed = candidate_mean >= pass_threshold
        won = v.winner == "A"
        wins += int(won)
        passes += int(passed)
        per_case.append(
            {
                "case_id": case_id,
                "winner": v.winner,
                "candidate_mean_score": round(candidate_mean, 3),
                "passed": passed,
                "scores": v.scores,
                "rationale": v.rationale,
            }
        )

    return {
        "config_label": config_label,
        "n_cases": n,
        "pass_rate": round(passes / n, 4) if n else 0.0,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "mean_scores_candidate": _mean_scores(all_verdicts, "A"),
        "mean_scores_baseline": _mean_scores(all_verdicts, "B"),
        "pass_threshold": pass_threshold,
        "cases": per_case,
    }


# ---------------------------------------------------------------------------
# A/B comparison report (two configs on the same suite)
# ---------------------------------------------------------------------------

def build_ab_report(
    config_a_label: str,
    config_a_verdicts: dict[str, Verdict],
    config_b_label: str,
    config_b_verdicts: dict[str, Verdict],
    pass_threshold: float = 6.0,
) -> dict[str, Any]:
    """Compare two configs run over the same suite and declare a winner.

    Each config's verdicts come from judging that config's output against
    the SAME baseline (or against each other, if you're comparing the two
    configs' outputs head-to-head — see README for which setup you used).
    The winner is declared on win_rate first, mean overall score as tiebreak.
    """
    report_a = build_suite_report(config_a_label, config_a_verdicts, pass_threshold)
    report_b = build_suite_report(config_b_label, config_b_verdicts, pass_threshold)

    mean_overall_a = statistics.mean(report_a["mean_scores_candidate"].values())
    mean_overall_b = statistics.mean(report_b["mean_scores_candidate"].values())

    if report_a["win_rate"] != report_b["win_rate"]:
        winner = config_a_label if report_a["win_rate"] > report_b["win_rate"] else config_b_label
        reason = (
            f"{winner} declared winner on win_rate "
            f"({report_a['win_rate']} vs {report_b['win_rate']})."
        )
    elif mean_overall_a != mean_overall_b:
        winner = config_a_label if mean_overall_a > mean_overall_b else config_b_label
        reason = (
            f"Win rates tied ({report_a['win_rate']}); {winner} declared winner on "
            f"tiebreak mean overall rubric score ({round(mean_overall_a, 3)} vs "
            f"{round(mean_overall_b, 3)})."
        )
    else:
        winner = "tie"
        reason = "Win rate and mean overall score are identical; no winner declared."

    return {
        "config_a": report_a,
        "config_b": report_b,
        "mean_overall_score": {config_a_label: round(mean_overall_a, 3), config_b_label: round(mean_overall_b, 3)},
        "declared_winner": winner,
        "justification": reason,
    }


# ---------------------------------------------------------------------------
# Position bias report
# ---------------------------------------------------------------------------

def build_position_bias_report(results: list[PositionBiasResult]) -> dict[str, Any]:
    """Summarize flip rate across a suite of both-orders judging runs."""
    n = len(results)
    flips = sum(1 for r in results if r.flipped)
    return {
        "n_cases": n,
        "flip_count": flips,
        "flip_rate": round(flips / n, 4) if n else 0.0,
        "cases": [
            {
                "case_id": r.case_id,
                "flipped": r.flipped,
                "winner_original_order": r.verdict_a_first.winner,
                "winner_swapped_order": r.verdict_b_first.winner,
                "final_winner": r.final_winner,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def write_report(report: dict[str, Any], filename: str) -> Path:
    """Write a report dict to reports/<filename> as pretty-printed JSON."""
    path = REPORTS_DIR / filename
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
