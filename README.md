# LLM Vulnerability-Triage Stability

A focused dependability study for **DESSERT 2026** (5–8 pp, IEEE). We test whether
off-the-shelf and open LLMs make **stable** vulnerability-triage decisions. We hold
the real risk signals fixed — CVSS (severity), EPSS (exploitation probability), CISA
KEV (already-exploited), asset context — and change **only the wording/order** of the
record in meaning-preserving ways. A competent analyst would give the same
`ACT / ATTEND / TRACK` verdict every time. We measure how often the LLM does not —
and, critically, how often it **downgrades** an urgent case.

We measure **decision stability, not accuracy.**

## Research question and contributions

> When the underlying risk signals are unchanged and only the presentation of a
> vulnerability record varies, do LLM triage decisions stay the same?

1. A released **meaning-preserving perturbation suite** over public CVE/EPSS/KEV records.
2. A **dependability evaluation** of LLM vulnerability triage that measures decision
   stability rather than accuracy.
3. **Safety-relevant stability metrics**: self-inconsistency, flip rate,
   safety-critical downgrade rate, over-escalation rate.

## Repository layout

```
.
├── pull_data.py          # build dataset.csv from NVD + EPSS + KEV
├── run_and_score.py      # API judges (OpenRouter) + scoring (--metrics-only to rescore)
├── run_local.py          # open-weights judges on a local GPU (same output schema)
├── inspect_flips.py      # qualitative before/after examples (Discussion)
├── plots.py              # regenerate figures/fig1..fig4 from the summary CSVs
├── requirements.txt
├── LICENSE
├── CLAUDE_PROJECT.md     # paste into a Claude Project for context
└── figures/              # generated figures
```

Generated artifacts (committed for reproducibility): `dataset.csv`,
`results_raw.csv`, `summary_flips.csv`, `summary_consistency.csv`,
`paraphrases.json`, `flips_examples.txt`, `figures/`.

## Setup

- Python 3.9+. The **API path needs no third-party packages**.
- For local models and plots: `pip install -r requirements.txt` (CUDA GPU for `run_local.py`).
- API key (one key, many models): `export OPENROUTER_API_KEY=...`
- Optional faster data pull: `export NVD_API_KEY=...`

## Reproduce (canonical sequence)

```bash
# 1. Build the dataset (commit dataset.csv and record the pull date below)
python pull_data.py --target-per-stratum 5 --out dataset_smoke.csv   # smoke test
python pull_data.py --target-per-stratum 80 --out dataset.csv

# 2. (Optional) freeze paraphrases so every model sees identical reworded text
python run_and_score.py --make-paraphrases --para-model openai/gpt-4o-mini

# 3. Run judges — start with a fast pilot, then scale
python run_and_score.py --limit 150 --repeats 3      # pilot (API models in MODELS)
python run_and_score.py --repeats 5                  # full (API models)
python run_local.py --model Qwen/Qwen2.5-7B-Instruct --repeats 5   # local model(s)

# 4. Score + figures + qualitative examples
python run_and_score.py --metrics-only
python plots.py
python inspect_flips.py --top 12
```

Edit the `MODELS` list in `run_and_score.py` and the policy thresholds (`POLICY`,
`gold_label`) before running. Seeds are fixed (`--seed`, default 13) for the data pull.

**Snapshot note (reproducibility):** EPSS and KEV update daily and NVD changes over
time, so commit the resulting `dataset.csv` and record when it was built:

> Data snapshot pulled on: `YYYY-MM-DD`.

## Data and ethics

Sources are public: NVD (U.S. government, public domain) for descriptions and CVSS;
FIRST EPSS for exploitation probability; CISA KEV for known-exploited status. The
released `dataset.csv` contains a CVE id, its public description, public scores, an
**assigned synthetic asset context**, and a **policy-derived label** (the gold label
is "correct under a stated policy," not universal truth — the policy is in
`run_and_score.py`). No private or sensitive data is used or redistributed.

## Cost

The API path cost scales as `calls × (~360 input + ~40 output tokens)`. A full
three-model run with one frontier API model is roughly **$15–20**; using only open
models locally is **≈ $0** (GPU time only). See `CLAUDE_PROJECT.md` / the paper notes
for the breakdown and cost knobs (fewer repeats, cheaper frontier, more local models).

## Citation

```bibtex
@inproceedings{tsangko2026triagestability,
  title     = {Same Vulnerability, Different Verdict: A Dependability Study of
               LLM-Based Vulnerability Triage Under Meaning-Preserving Perturbations},
  author    = {Tsangko, Iosif and others},
  booktitle = {Proc. IEEE Int. Conf. Dependable Systems, Services and Technologies (DESSERT)},
  year      = {2026}
}
```
