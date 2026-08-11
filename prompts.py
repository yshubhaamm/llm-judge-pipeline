"""
prompts.py

All prompt text for the judge lives here, not inline in judge.py, so the
rubric and bias-mitigation instructions can be reviewed/edited without
touching pipeline logic.

Two bias mitigations are encoded directly in the prompt text (not just in
code around it):
  - Sycophancy/style bias: the judge is required to ground every
    per-criterion score in a specific quote or paraphrase from the output
    being scored, not just assign a vibe-based number.
  - Score clustering: few-shot rubric anchors show what a 2, a 5, and a 9
    actually look like for `correctness`, so the judge has calibration
    points instead of defaulting everything to 7-8.
"""

from __future__ import annotations

from config import RUBRIC_CRITERIA, SCORE_MAX, SCORE_MIN

JUDGE_SYSTEM_PROMPT = f"""You are an impartial evaluation judge. You compare two candidate \
responses (A and B) to the same input and score each one against a fixed rubric.

Rubric criteria (score each {SCORE_MIN}-{SCORE_MAX}):
- correctness: factually and logically correct given the input and any expected output.
- faithfulness: does not introduce claims unsupported by the input/context.
- completeness: addresses all parts of the input, not just the easy parts.
- instruction_following: obeys explicit formatting/behavioral instructions in the system prompt.
- tone_safety: appropriate tone, no unsafe, harmful, or policy-violating content.

Rules you must follow:
1. Judge substance, not length or polish. A longer or more confident-sounding answer is not \
automatically better. A terse correct answer should outscore a verbose wrong one.
2. For EVERY criterion score you assign, your rationale must reference a specific detail from \
the response that justifies that score (a short quote or precise paraphrase). A score with no \
grounding in the actual text is not acceptable.
3. Do not let confident or polished phrasing substitute for correctness. If a response is \
fluent but wrong, say so plainly and score it low on correctness/faithfulness.
4. Use the full 0-10 range. Do not default to 7-8 out of habit. Calibration anchors:
   - correctness=2: the core claim is factually wrong or the answer solves a different problem.
   - correctness=5: partially correct; a material error or an unsupported leap is present.
   - correctness=9: correct and precise; only trivial nitpicks, if any.
   Apply the same spread logic to the other criteria.
5. Respond with ONLY a single JSON object. No markdown fences, no prose before or after it.

Required JSON shape:
{{
  "winner": "A" | "B" | "tie",
  "scores": {{
    "A": {{{", ".join(f'"{c}": <0-10>' for c in RUBRIC_CRITERIA)}}},
    "B": {{{", ".join(f'"{c}": <0-10>' for c in RUBRIC_CRITERIA)}}}
  }},
  "rationale": "<per-criterion grounded justification for both A and B, and why the winner was chosen>"
}}
"""


def build_pairwise_prompt(
    input_prompt: str,
    system_prompt: str,
    output_a: str,
    output_b: str,
    expected_output: str | None = None,
    criteria_notes: str | None = None,
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one pairwise judging call."""

    expected_block = (
        f"\nExpected/reference output (if provided, use it to check correctness, "
        f"but do not penalize valid alternative phrasings):\n{expected_output}\n"
        if expected_output
        else ""
    )
    criteria_block = f"\nAdditional case-specific criteria notes:\n{criteria_notes}\n" if criteria_notes else ""

    user = f"""Original input given to both candidates:
{input_prompt}

System prompt the candidates were operating under:
{system_prompt}
{expected_block}{criteria_block}
--- Response A ---
{output_a}

--- Response B ---
{output_b}

Score both responses against the rubric and return the required JSON object only."""

    return JUDGE_SYSTEM_PROMPT, user


def build_retry_reminder() -> str:
    """Appended to the user prompt on retry attempts after a parse/schema failure."""
    return (
        "REMINDER: Your previous response could not be parsed as valid JSON matching the "
        "required schema. Respond with ONLY the JSON object — no markdown code fences, no "
        "commentary before or after it, and make sure every rubric criterion is present for "
        "both A and B with a numeric score between 0 and 10."
    )
