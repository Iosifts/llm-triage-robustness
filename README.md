# LLM Vulnerability-Triage Stability

This repository contains a controlled benchmark for evaluating whether LLMs make stable vulnerability-triage decisions under meaning-preserving prompt perturbations.

The setup holds the underlying risk signals fixed — CVSS severity, EPSS exploitation probability, CISA KEV status, and asset context — while changing only the wording or ordering of the vulnerability record. The goal is to measure whether the model gives the same `ACT / ATTEND / TRACK` verdict when the substantive information is unchanged.

We measure decision stability, not universal triage correctness.

## Project goal

When the underlying risk signals are unchanged and only the presentation of a vulnerability record varies, do LLM triage decisions stay the same?

This project includes:

1. A meaning-preserving perturbation suite over public CVE, EPSS, and KEV records.
2. Scripts for evaluating LLM vulnerability-triage stability.
3. Stability metrics including self-inconsistency, flip rate, downgrade rate, and over-escalation rate.

## Repository layout

```text
.
├── pull_data.py          # build dataset.csv from NVD + EPSS + KEV
├── run_and_score.py      # API judges via OpenRouter + scoring
├── run_local.py          # open-weight judges on a local GPU
├── inspect_flips.py      # qualitative before/after examples
├── plots.py              # regenerate figures from summary CSVs
├── requirements.txt
├── LICENSE
├── CLAUDE_PROJECT.md     # optional project context notes
└── figures/              # generated figures, if present
```

Generated artifacts may include:

```text
dataset.csv
results_full.csv
summary_flips.csv
summary_consistency.csv
stats_rates_ci.csv
stats_reorder_tests.csv
paraphrases.json
flips_examples.txt
figures/
```

## Setup

Python 3.9+ is recommended.

The API path uses OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

Optional faster NVD data pulls:

```bash
export NVD_API_KEY=...
```

Install dependencies for local models, plotting, and analysis:

```bash
pip install -r requirements.txt
```

Local model runs require a suitable CUDA GPU.

## Reproduce

Build the dataset:

```bash
python pull_data.py --target-per-stratum 5 --out dataset_smoke.csv
python pull_data.py --target-per-stratum 80 --out dataset.csv
```

Optionally freeze paraphrases so every model sees identical reworded records:

```bash
python run_and_score.py --make-paraphrases --para-model openai/gpt-4o-mini
```

Run API-based judges:

```bash
python run_and_score.py --limit 150 --repeats 3
python run_and_score.py --repeats 5
```

Run a local open-weight model:

```bash
python run_local.py --model Qwen/Qwen2.5-7B-Instruct --repeats 5
```

Score results and regenerate analysis artifacts:

```bash
python run_and_score.py --metrics-only
python plots.py
python inspect_flips.py --top 12
```

Edit the `MODELS` list in `run_and_score.py` and the policy thresholds in `POLICY` / `gold_label` before running new experiments.

Seeds are fixed by default with `--seed 13` for the data pull.

## Snapshot note

EPSS and KEV update over time, and NVD records may change. For reproducibility, commit the generated `dataset.csv` and record when it was built:

```text
Data snapshot pulled on: YYYY-MM-DD
```

## Data and ethics

The project uses public vulnerability data:

* NVD for CVE descriptions and CVSS scores
* FIRST EPSS for exploitation probability
* CISA KEV for known-exploited status

The generated `dataset.csv` contains public CVE information, public risk scores, assigned synthetic asset context, and a policy-derived label. The label is correct under the stated policy in `run_and_score.py`; it is not intended to be a universal ground truth for vulnerability management.

No private or sensitive data is used or redistributed.

## Cost

API cost depends on the number of model calls and token usage. The main knobs are:

* number of models
* number of records
* number of perturbations
* number of repeats
* choice of API model

Local open-weight models avoid API costs but require GPU time.
