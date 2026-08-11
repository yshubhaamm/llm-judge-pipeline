"""
validate.py

Three independent forms of evidence that the judge is trustworthy, per the
assignment's validation requirement:

1. Agreement with human labels (agreement rate + Cohen's kappa) — are the
   judge's A/B/tie calls consistent with what a person would say?
2. Test-retest consistency — run the same case through the judge twice
   (same order) and report how often the verdict changes with nothing
   about the inputs having changed. This is noise, not position bias.
3. Adversarial probes — deliberately hard cases (verbose-but-wrong,
   terse-but-correct, confidently-wrong, polished-but-wrong) where we
   already know the "correct" answer and check whether the judge is fooled.

Cohen's kappa is implemented from scratch (no sklearn dependency) since
the label set is small (3 classes: A/B/tie) and the formula is simple.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from config import PipelineConfig
from judge import CaseInput, PairwiseJudge


LABELS = ("A", "B", "tie")


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Compute Cohen's kappa for two raters over a fixed label set.

    kappa = (p_observed - p_expected) / (1 - p_expected)
    where p_expected is derived from each rater's marginal label
    frequencies under the assumption of independence.
    """
    n = len(rater_a)
    if n == 0 or n != len(rater_b):
        raise ValueError("Both rating lists must be the same non-zero length.")

    p_observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    p_expected = sum(
        (counts_a.get(label, 0) / n) * (counts_b.get(label, 0) / n) for label in LABELS
    )

    if p_expected == 1.0:
        # Perfect agreement expected by chance alone (degenerate label set);
        # kappa is undefined, but observed agreement is informative on its own.
        return 1.0 if p_observed == 1.0 else 0.0

    return round((p_observed - p_expected) / (1 - p_expected), 4)


def _case_from_row(row: dict[str, Any]) -> CaseInput:
    return CaseInput(
        case_id=row["case_id"],
        input_prompt=row["input_prompt"],
        system_prompt=row.get("system_prompt", ""),
        output_a=row["output_a"],
        output_b=row["output_b"],
        expected_output=row.get("expected_output"),
        criteria_notes=row.get("criteria_notes"),
    )


def run_agreement_check(judge: PairwiseJudge, human_labels: list[dict[str, Any]]) -> dict[str, Any]:
    judge_calls: list[str] = []
    human_calls: list[str] = []
    per_case = []

    for row in human_labels:
        case = _case_from_row(row)
        verdict = judge.judge_pair(case)
        judge_calls.append(verdict.winner)
        human_calls.append(row["human_winner"])
        per_case.append(
            {
                "case_id": case.case_id,
                "human_winner": row["human_winner"],
                "judge_winner": verdict.winner,
                "agree": verdict.winner == row["human_winner"],
            }
        )

    agreement_rate = round(sum(1 for r in per_case if r["agree"]) / len(per_case), 4) if per_case else 0.0
    kappa = cohens_kappa(judge_calls, human_calls) if per_case else 0.0

    return {
        "n_cases": len(per_case),
        "agreement_rate": agreement_rate,
        "cohens_kappa": kappa,
        "cases": per_case,
    }


def run_test_retest(judge: PairwiseJudge, human_labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge each labeled case twice, same order, and report the flip rate."""
    flips = 0
    per_case = []
    for row in human_labels:
        case = _case_from_row(row)
        first = judge.judge_pair(case)
        second = judge.judge_pair(case)
        flipped = first.winner != second.winner
        flips += int(flipped)
        per_case.append(
            {
                "case_id": case.case_id,
                "first_winner": first.winner,
                "second_winner": second.winner,
                "flipped": flipped,
            }
        )
    n = len(per_case)
    return {
        "n_cases": n,
        "flip_count": flips,
        "flip_rate": round(flips / n, 4) if n else 0.0,
        "cases": per_case,
    }


def run_adversarial_probes(judge: PairwiseJudge, probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Each probe declares which side (A/B) is actually correct via `expected_winner`.

    A probe is "fooled" if the judge's winner does not match expected_winner.
    Probe `probe_type` values (for readability in the report, not used in logic):
    verbose_but_wrong, terse_but_correct, confidently_wrong, polished_but_wrong.
    """
    fooled = 0
    per_probe = []
    for row in probes:
        case = _case_from_row(row)
        verdict = judge.judge_pair(case)
        was_fooled = verdict.winner != row["expected_winner"]
        fooled += int(was_fooled)
        per_probe.append(
            {
                "case_id": case.case_id,
                "probe_type": row.get("probe_type", "unspecified"),
                "expected_winner": row["expected_winner"],
                "judge_winner": verdict.winner,
                "fooled": was_fooled,
            }
        )
    n = len(per_probe)
    return {
        "n_probes": n,
        "fooled_count": fooled,
        "fool_rate": round(fooled / n, 4) if n else 0.0,
        "probes": per_probe,
    }


def run_validation(
    pipeline_config: PipelineConfig, labels_path: Path, probes_path: Path
) -> dict[str, Any]:
    human_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    probes = json.loads(probes_path.read_text(encoding="utf-8"))

    judge = PairwiseJudge(pipeline_config)

    agreement = run_agreement_check(judge, human_labels)
    test_retest = run_test_retest(judge, human_labels)
    adversarial = run_adversarial_probes(judge, probes)

    return {
        "agreement": agreement,
        "test_retest": test_retest,
        "adversarial": adversarial,
        "token_usage": judge.token_tracker.summary(),
    }
