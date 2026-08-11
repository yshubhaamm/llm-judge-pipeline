"""
main.py

Command-line entry point for the pipeline.

Usage:
    python main.py evaluate --suite suites/suite.json --label baseline
    python main.py compare  --suite suites/suite.json --config-a suites/config_a.json --config-b suites/config_b.json
    python main.py validate --labels validation/human_labels.json --probes validation/adversarial_probes.json

See README.md for full walkthroughs of each mode.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config import load_pipeline_config, SUITES_DIR, VALIDATION_DIR, PipelineConfig
from judge import CaseInput, PairwiseJudge, PositionBiasResult
from parser import Verdict
from report import build_ab_report, build_position_bias_report, build_suite_report, write_report
from validate import run_validation


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cases_from_suite(suite_data: list[dict[str, Any]]) -> list[CaseInput]:
    cases = []
    for row in suite_data:
        cases.append(
            CaseInput(
                case_id=row["case_id"],
                input_prompt=row["input_prompt"],
                system_prompt=row.get("system_prompt", ""),
                output_a=row["output_a"],
                output_b=row["output_b"],
                expected_output=row.get("expected_output"),
                criteria_notes=row.get("criteria_notes"),
            )
        )
    return cases


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Judge every case in a suite, both orders, and write suite + position-bias reports."""
    pipeline_config = load_pipeline_config()
    suite_path = Path(args.suite)
    suite_data = _load_json(suite_path)
    cases = _cases_from_suite(suite_data)

    judge = PairwiseJudge(pipeline_config)

    verdicts: dict[str, Verdict] = {}
    position_results: list[PositionBiasResult] = []

    for case in cases:
        if pipeline_config.run_both_orders:
            pb_result = judge.judge_pair_both_orders(case)
            position_results.append(pb_result)
            verdicts[case.case_id] = pb_result.verdict_a_first
        else:
            verdicts[case.case_id] = judge.judge_pair(case)

    suite_report = build_suite_report(args.label, verdicts, pass_threshold=args.pass_threshold)
    suite_report["token_usage"] = judge.token_tracker.summary()
    write_report(suite_report, "suite_report.json")
    print(f"suite_report.json written — pass_rate={suite_report['pass_rate']} "
          f"win_rate={suite_report['win_rate']}")

    if position_results:
        pb_report = build_position_bias_report(position_results)
        write_report(pb_report, "position_bias_report.json")
        print(f"position_bias_report.json written — flip_rate={pb_report['flip_rate']} "
              f"({pb_report['flip_count']}/{pb_report['n_cases']})")


def cmd_compare(args: argparse.Namespace) -> None:
    """Judge the same suite under two configs and declare a winner."""
    pipeline_config = load_pipeline_config()
    suite_data = _load_json(Path(args.suite))
    cases = _cases_from_suite(suite_data)

    config_a_meta = _load_json(Path(args.config_a))
    config_b_meta = _load_json(Path(args.config_b))

    judge = PairwiseJudge(pipeline_config)

    verdicts_a: dict[str, Verdict] = {}
    verdicts_b: dict[str, Verdict] = {}
    for case in cases:
        verdicts_a[case.case_id] = judge.judge_pair(case)
        # Config B's "candidate" output lives in output_b for these cases;
        # swap so build_suite_report's "side A = candidate" convention holds.
        swapped_case = CaseInput(
            case_id=case.case_id,
            input_prompt=case.input_prompt,
            system_prompt=case.system_prompt,
            output_a=case.output_b,
            output_b=case.output_a,
            expected_output=case.expected_output,
            criteria_notes=case.criteria_notes,
        )
        verdicts_b[case.case_id] = judge.judge_pair(swapped_case)

    ab_report = build_ab_report(
        config_a_meta.get("label", "config_a"),
        verdicts_a,
        config_b_meta.get("label", "config_b"),
        verdicts_b,
        pass_threshold=args.pass_threshold,
    )
    ab_report["token_usage"] = judge.token_tracker.summary()
    write_report(ab_report, "ab_report.json")
    print(f"ab_report.json written — declared_winner={ab_report['declared_winner']}")
    print(ab_report["justification"])


def cmd_validate(args: argparse.Namespace) -> None:
    """Run agreement, test-retest, and adversarial-probe validation."""
    pipeline_config = load_pipeline_config()
    labels_path = Path(args.labels)
    probes_path = Path(args.probes)
    result = run_validation(pipeline_config, labels_path, probes_path)
    out_path = VALIDATION_DIR / "validation_report.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"validation_report.json written to {out_path}")
    print(f"agreement_rate={result['agreement']['agreement_rate']} "
          f"cohens_kappa={result['agreement']['cohens_kappa']}")
    print(f"test_retest_flip_rate={result['test_retest']['flip_rate']}")
    print(f"adversarial_fool_rate={result['adversarial']['fool_rate']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser("evaluate", help="Judge a suite and write suite_report.json")
    p_eval.add_argument("--suite", default=str(SUITES_DIR / "suite.json"))
    p_eval.add_argument("--label", default="default")
    p_eval.add_argument("--pass-threshold", type=float, default=6.0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_compare = subparsers.add_parser("compare", help="Compare two configs, write ab_report.json")
    p_compare.add_argument("--suite", default=str(SUITES_DIR / "suite.json"))
    p_compare.add_argument("--config-a", default=str(SUITES_DIR / "config_a.json"))
    p_compare.add_argument("--config-b", default=str(SUITES_DIR / "config_b.json"))
    p_compare.add_argument("--pass-threshold", type=float, default=6.0)
    p_compare.set_defaults(func=cmd_compare)

    p_validate = subparsers.add_parser("validate", help="Run judge validation suite")
    p_validate.add_argument("--labels", default=str(VALIDATION_DIR / "human_labels.json"))
    p_validate.add_argument("--probes", default=str(VALIDATION_DIR / "adversarial_probes.json"))
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # top-level guard so CLI failures are readable
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
