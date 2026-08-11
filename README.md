# LLM-as-Judge Evaluation Pipeline

A pairwise (A-vs-B) LLM evaluation pipeline built for the Nexpro AI take-home assignment (Problem 2). The pipeline takes a test case containing an input prompt, a system prompt, and two candidate model outputs, then produces a structured verdict, measures common judge biases, and validates the judge against human-labeled examples and adversarial probes.

## What I built

* Pairwise A-vs-B judging with a structured JSON verdict
* Rubric-based scoring across five criteria
* Position bias detection using A→B and B→A evaluation
* A/B comparison between two prompt configurations
* Validation against human labels
* Adversarial probe testing
* JSON reports for reproducible evaluation

## Models used

* **Judge:** Gemini 3.5 Flash-Lite
* **Generator:** OpenAI GPT-4o-mini

The judge and generator are configured independently through environment variables so that the evaluator is not tied to the same model family that produced the candidate outputs.

## Rubric

Each response is scored on a 0-10 scale for:

* Correctness
* Faithfulness
* Completeness
* Instruction-following
* Tone / safety

The judge is instructed to justify scores using evidence from the candidate responses rather than style or verbosity alone.

## Architecture

```text
suites/*.json
      |
      v
prompts.py  ->  judge.py  ->  Gemini judge model
      |             |
      |             v
      |        parser.py
      |             |
      v             v
      +-------> report.py
                     |
                     +--> suite_report.json
                     +--> ab_report.json
                     +--> position_bias_report.json

validate.py
      |
      +--> validation_report.json
```

The entry point is `main.py`, which supports `evaluate`, `compare`, and `validate`.

## Project structure

```text
llm-judge-pipeline/
├── main.py
├── judge.py
├── parser.py
├── prompts.py
├── report.py
├── validate.py
├── config.py
├── utils.py
├── suites/
├── reports/
├── validation/
├── logs/
└── README.md
```

## Setup

### Requirements

* Python 3.10+
* Gemini API key
* OpenAI API key

### Installation

```bash
git clone https://github.com/yshubhaamm/llm-judge-pipeline.git
cd llm-judge-pipeline

python -m venv venv
venv\\Scripts\\activate        # Windows

pip install -r requirements.txt
```

Create a `.env` file using the same variable names as `.env.example`.

### Example configuration

```text
JUDGE_PROVIDER=gemini
JUDGE_MODEL=gemini-3.5-flash-lite

GENERATOR_PROVIDER=openai
GENERATOR_MODEL=gpt-4o-mini

RUN_BOTH_ORDERS=true
```

## Running the pipeline

### Evaluate a test suite

```bash
python main.py evaluate --suite suites/suite.json --label final
```

Outputs:

* `reports/suite_report.json`
* `reports/position_bias_report.json`

### Compare two prompt configurations

```bash
python main.py compare \
  --suite suites/suite.json \
  --config-a suites/config_a.json \
  --config-b suites/config_b.json
```

Output:

* `reports/ab_report.json`

### Run validation

```bash
python main.py validate \
  --labels validation/human_labels.json \
  --probes validation/adversarial_probes.json
```

Output:

* `validation/validation_report.json`

## Evaluation results

The final run used **5 evaluation cases** covering arithmetic, grounding, instruction following, safety, and completeness.

### Suite evaluation

| Metric    | Result  |
| --------- | ------- |
| Pass rate | **1.0** |
| Win rate  | **1.0** |

### Position bias

The pipeline evaluates every pair twice (A→B and B→A) and compares the results.

| Metric          | Result  |
| --------------- | ------- |
| Cases evaluated | **5**   |
| Flip count      | **0**   |
| Flip rate       | **0.0** |

This indicates that the judge was stable with respect to response ordering on the evaluation suite.

### Judge validation

Validation was performed on **12 human-labeled cases**.

| Metric                | Result           |
| --------------------- | ---------------- |
| Agreement rate        | **1.0**          |
| Cohen’s kappa         | **1.0**          |
| Test-retest flip rate | **0.0**          |
| Adversarial fool rate | **0.1667 (1/6)** |

The judge matched all human labels in the validation set and remained consistent across repeated evaluations.

### A/B comparison

| Configuration                  | Pass rate | Win rate |
| ------------------------------ | --------- | -------- |
| `prompt_v1_concise`            | **1.0**   | **1.0**  |
| `prompt_v2_explicit_grounding` | **0.4**   | **0.0**  |

**Winner:** `prompt_v1_concise`

## Design decisions

### Why pairwise instead of pointwise?

I chose pairwise A-vs-B judging because it was easier to make consistent across runs. The judge only needs to decide which response is better instead of assigning an absolute score, which reduced score drift during repeated evaluations.

### Position bias handling

For every case, the pipeline judges both **A→B** and **B→A**. The swapped verdict is mapped back to the original candidate labels before aggregation. The measured flip rate is reported separately.

### Structured parsing

The judge is asked to return JSON. If the response is malformed, the parser attempts recovery before the request is retried. This was useful because Gemini occasionally returned partially formatted JSON during early testing.

## Limitations

* The quality of the evaluation still depends on the judge model.
* The adversarial probe set is small and does not cover every possible failure mode.
* Position-bias checks measure consistency, not correctness.
* Human validation is still important for subjective tasks.

## What I would improve next

* Add support for multiple judge models and ensemble voting
* Expand the adversarial probe suite
* Add confidence calibration and disagreement analysis
* Build a small dashboard for report visualization

## Repository

GitHub: https://github.com/yshubhaamm/llm-judge-pipeline
