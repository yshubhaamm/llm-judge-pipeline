# LLM-as-Judge Evaluation Pipeline

A pairwise (A-vs-B) LLM judging pipeline built for Nexpro AI's take-home
assignment (Problem 2). Given a test case — an input, a system prompt, and
two candidate outputs — the pipeline produces a structured, rubric-based
verdict, measures the judge's own biases, and validates the judge against
human labels and adversarial probes.

## Overview

- **Judging mode:** pairwise A-vs-B. Chosen over pointwise scoring because
  pairwise comparisons are more robust to score-scale drift between runs —
  the judge only has to say which is better, not agree with its past self
  on what an absolute "7" means. See [Design decisions](#design-decisions--trade-offs)
  for the full trade-off discussion.
- **Rubric:** correctness, faithfulness, completeness, instruction_following,
  tone_safety — each scored 0-10 per side, with a mandatory grounding
  requirement (see `prompts.py`).
- **Bias handling:** position bias (both-orders runs), verbosity bias
  (adversarial probes), self-enhancement bias (independently configurable
  judge/generator model families), sycophancy bias (forced grounding),
  score clustering (few-shot calibration anchors in the prompt).
- **Validation:** agreement + Cohen's kappa against human labels,
  test-retest consistency, adversarial probe fooling rate.

## Architecture

```
                         ┌─────────────────┐
                         │   suites/*.json  │   test cases: input,
                         │  (test suites)   │   system prompt, output A/B
                         └────────┬─────────┘
                                  │
                                  ▼
┌────────────┐          ┌──────────────────┐          ┌─────────────────┐
│ prompts.py │─────────▶│    judge.py       │◀────────▶│  LLM Provider    │
│ (rubric +  │  builds  │  PairwiseJudge     │  calls   │ (Anthropic /     │
│ grounding  │  prompt  │  - judge_pair()    │          │  OpenAI, via     │
│ + anchors) │          │  - both_orders()   │          │  config.py)      │
└────────────┘          └────────┬──────────┘          └─────────────────┘
                                  │ raw text response
                                  ▼
                         ┌──────────────────┐
                         │   parser.py       │   strip fences → direct
                         │  robust JSON       │   parse → regex extract →
                         │  extraction        │   repair → validate schema
                         └────────┬──────────┘
                                  │ Verdict (or retry on failure)
                                  ▼
                         ┌──────────────────┐
                         │   utils.py         │   JSONL logs (prompts +
                         │  logging + tokens   │   responses), token/cost
                         └────────┬──────────┘   tracking
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   report.py        │   suite_report.json
                         │  aggregation        │   ab_report.json
                         │                     │   position_bias_report.json
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   validate.py       │   validation_report.json
                         │  human agreement,    │   (agreement, kappa,
                         │  test-retest,        │    test-retest, probes)
                         │  adversarial probes  │
                         └──────────────────┘

Entry point for all of the above: main.py (evaluate / compare / validate)
```

## Setup

### Requirements
- Python 3.10+
- An Anthropic API key and/or OpenAI API key, depending on which
  provider(s) you configure for judge/generator.

### Installation

```bash
git clone <your-repo-url>
cd llm-judge-pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
```

Load the `.env` file before running (either `export $(cat .env | xargs)`
on Linux/macOS, or add `from dotenv import load_dotenv; load_dotenv()`
at the top of `main.py` if you'd rather not manage shell exports).

### Environment variables

| Variable | Purpose |
|---|---|
| `JUDGE_PROVIDER` | `anthropic`, `openai`, or `gemini` — which SDK to use for the judge |
| `JUDGE_MODEL` | Model name for the judge |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | API keys (only the one matching your chosen provider(s) is required) |
| `JUDGE_TEMPERATURE`, `JUDGE_MAX_TOKENS` | Judge sampling params |
| `GENERATOR_PROVIDER`, `GENERATOR_MODEL` | Same, for whichever model produced the outputs being judged |
| `*_COST_PER_MILLION` | Optional, for the cost estimator in reports |
| `MAX_RETRIES`, `RETRY_BACKOFF_SECONDS` | Retry behavior on malformed judge output |
| `RUN_BOTH_ORDERS` | `true`/`false` — whether `evaluate` runs position-bias checks by default |

**Self-enhancement bias mitigation:** set `JUDGE_PROVIDER`/`JUDGE_MODEL` and
`GENERATOR_PROVIDER`/`GENERATOR_MODEL` to genuinely different model families —
e.g. Gemini as judge, OpenAI as generator (the shipped `.env.example` default) —
rather than two models from the same provider (a same-provider pairing like
GPT-4o judging GPT-4o-mini is weaker evidence against self-enhancement bias,
since sibling models still share training data and stylistic preferences).

## Running evaluation

```bash
python main.py evaluate --suite suites/suite.json --label my_config
```

This judges every case in the suite, runs both A-vs-B and B-vs-A orderings
(position-bias check), and writes:
- `reports/suite_report.json` — pass rate, win rate, mean scores per criterion
- `reports/position_bias_report.json` — flip count / flip rate

## Running an A/B comparison

```bash
python main.py compare \
  --suite suites/suite.json \
  --config-a suites/config_a.json \
  --config-b suites/config_b.json
```

Each case in the suite is expected to carry both configs' candidate outputs
(`output_a` = config A's output, `output_b` = config B's output for that
case). The command judges each config's output against the other and
writes `reports/ab_report.json`, including a declared winner and a
one-line justification (win rate first, mean rubric score as tiebreak).

## Running validation

```bash
python main.py validate \
  --labels validation/human_labels.json \
  --probes validation/adversarial_probes.json
```

Writes `validation/validation_report.json` containing:
- **Agreement**: judge-vs-human agreement rate and Cohen's kappa
- **Test-retest**: how often the judge's verdict changes on an identical
  re-run (same order, same inputs)
- **Adversarial probes**: fool rate on verbose-but-wrong, terse-but-correct,
  confidently-wrong, and polished-but-wrong probes

> `validation/human_labels.json` ships with a starter set of 12 cases where
> the "correct" side is objectively determinable (arithmetic, exact
> context-grounding, explicit format rules) so the labels are honestly
> defensible as-is. If you have your own human-reviewed preference data,
> swap it in — that's stronger evidence than objectively-verifiable cases,
> which are a good regression check but a lower bar than genuine subjective
> human judgment.

## Report generation

All three `main.py` subcommands write their reports directly — there's no
separate report-generation step. Reports are plain JSON so they're easy to
diff between runs or pull into the submission document.

## Design decisions & trade-offs

- **Pairwise over pointwise.** Pointwise scoring (rate this one output 0-10)
  is faster and cheaper per case, but absolute scores drift across
  sessions and models cluster scores in a narrow band without a comparison
  anchor. Pairwise forces a relative judgment, which is more stable and is
  what the assignment's bias-mitigation table (position bias, in
  particular) assumes.
- **Retry at the parse layer, not the network layer.** `judge.py` retries
  when `parser.py` can't extract a valid verdict, not on network errors —
  those are different failure modes and conflating them would mean
  retrying a rate-limit error with a "please output valid JSON" reminder,
  which doesn't help.
- **Position-bias remapping.** When a case is judged in swapped order, the
  verdict's A/B labels are remapped back to the caller's original A/B
  *before* being handed to any other module. Every downstream consumer
  (report.py, validate.py) can assume "A" always means the same thing,
  which keeps aggregation logic simple.
- **JSONL logging, not a database.** Judge prompts/responses are logged as
  append-only JSON lines specifically so a run can be replayed or audited
  by reading the file top to bottom — no schema migrations, no query
  layer needed for a take-home-sized suite.
- **Cohen's kappa implemented from scratch.** The label set is small (A/B/tie)
  and the formula is a few lines; pulling in scikit-learn for one function
  isn't worth the dependency weight.

## Bias handling summary

| Bias | Mitigation | Where |
|---|---|---|
| Position | Every `evaluate` run judges both A-vs-B and B-vs-A; flip rate reported | `judge.judge_pair_both_orders`, `position_bias_report.json` |
| Verbosity | Adversarial probes with verbose-wrong / terse-correct pairs; prompt explicitly instructs the judge to ignore length | `prompts.py` rule 1, `validation/adversarial_probes.json` |
| Self-enhancement | Judge and generator configured via fully independent env vars/providers | `config.py` |
| Sycophancy / style | Judge is required to ground every score in a specific quote/paraphrase; confidently-wrong probes included | `prompts.py` rule 2, `probe_confidently_wrong_01` |
| Score clustering | Few-shot rubric anchors (2/5/9 examples) in the judge system prompt | `prompts.py` rule 4 |

**Would I let this gate a release?** For low-stakes, high-volume regression
checks (e.g. did a prompt tweak break formatting compliance), yes — the
position-bias and adversarial-probe checks give reasonable confidence the
judge isn't trivially foolable. For anything safety-critical or where a
false pass has real cost, I'd treat the judge's verdict as a first-pass
filter and route flagged or borderline cases (ties, high position-bias flip
cases, low-margin win rates) to human review rather than auto-gating on it.

## Limitations

- The judge is only as good as its own model's judgment — it can still be
  fooled by adversarial cases outside the probe set's specific patterns.
- Cost/latency estimates depend on the `*_COST_PER_MILLION` values you set
  in `.env`; the defaults in `.env.example` are placeholders, not live
  pricing, and will go stale — check current provider pricing before
  relying on the cost estimate.
- `compare` assumes both configs' outputs already exist in the suite file;
  this pipeline does not itself call generator models to produce fresh
  outputs per config (that's a natural extension — see `ChatClient` in
  `judge.py`, which the generator side could reuse).
- Test-retest and position-bias checks measure *consistency*, not
  *correctness* — a judge can be perfectly consistent and still wrong.
  Agreement-with-humans is the check that catches that, which is why both
  are reported separately rather than collapsed into one score.

## When human review is still required

- Any case where `flipped: true` in the position-bias report — the judge's
  own preference wasn't stable, so its verdict shouldn't be trusted as-is.
- Cases the adversarial probe set structurally can't cover (novel failure
  modes, domain-specific correctness the judge has no grounding to check).
- Safety-relevant verdicts in general — the `tone_safety` criterion is a
  useful signal, not a substitute for a human safety review before
  anything user-facing ships.
- Low human-agreement domains — if a future validation run against a
  larger human-labeled set shows kappa below a reasonable threshold
  (rule of thumb: below ~0.4, "fair" agreement or worse) for a given case
  category, that category's verdicts need human spot-checking until the
  rubric or prompt is improved.
