#!/usr/bin/env python3
"""
STEP 2 - Perturb (meaning-preserving), query LLM judges, parse verdicts, score.

Conditions change ONLY the wording/order. CVSS, EPSS, KEV, asset are always identical.
  original        - baseline
  reorder         - same fields, different order
  add_lowstakes   - prepend a benign low-stakes background sentence
  add_highstakes  - prepend a benign high-stakes background sentence
  compress        - keep only the first 1-2 sentences of the description
  expand          - append a benign, non-informative sentence
  paraphrase      - (optional) LLM-reworded description; only if paraphrases.json exists

Outputs results_raw.csv (long format, one row per cve x model x condition x repeat):
  cve_id,stratum,asset,gold_label,model,condition,repeat,parsed_verdict,parse_ok,raw_output

Modes:
  export OPENROUTER_API_KEY=xxxx
  python run_and_score.py --make-paraphrases --para-model openai/gpt-4o-mini   # one-time
  python run_and_score.py --dataset dataset.csv --repeats 5                    # run + score
  python run_and_score.py --metrics-only                                       # rescore only

Start SMALL: --limit 150 --repeats 3  (get a signal fast, then scale up).
Supports resume: re-running skips (cve,model,condition,repeat) already in results_raw.csv.
Edit MODELS below: one frontier + one mid + one open is ideal.
"""
import argparse, csv, json, os, re, sys, time, urllib.request
from collections import defaultdict, Counter

MODELS = [
    "openai/gpt-4o",                      # frontier  (swap as you like)
    "anthropic/claude-3.5-haiku",         # mid / cheap
    "meta-llama/llama-3.1-8b-instruct",   # open
]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY", "")

VAL = {"ACT": 2, "ATTEND": 1, "TRACK": 0}
LOWSTAKES  = "Background note: this product is used mainly by hobbyists for small personal projects."
HIGHSTAKES = "Background note: this product is widely deployed across large enterprise environments."
EXPAND     = "For additional context, administrators may wish to consult vendor documentation and follow standard operational procedures where applicable."

POLICY = (
    "Use this triage policy:\n"
    "- ACT: the vulnerability is on the CISA KEV list, OR EPSS >= 0.50 on a critical/internet-facing asset, "
    "OR CVSS >= 9.0 with EPSS >= 0.30.\n"
    "- ATTEND: CVSS >= 7.0 with EPSS >= 0.10, OR EPSS >= 0.30, OR a critical asset with CVSS >= 7.0.\n"
    "- TRACK: none of the above.\n"
)
SYSTEM = "You are a vulnerability triage assistant. Apply the given policy exactly and consistently."
ORDER_A = ["Description", "CVSS base score", "EPSS exploitation probability", "Known exploited (CISA KEV)", "Affected asset"]
ORDER_B = ["Affected asset", "Known exploited (CISA KEV)", "EPSS exploitation probability", "CVSS base score", "Description"]
CONDITIONS = ["original", "reorder", "add_lowstakes", "add_highstakes", "compress", "expand"]

def first_sentences(text, n=2):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()

def desc_for(row, condition, paraphrases):
    base = row["description"]
    if condition == "add_lowstakes":  return LOWSTAKES + " " + base
    if condition == "add_highstakes": return HIGHSTAKES + " " + base
    if condition == "compress":       return first_sentences(base, 2)
    if condition == "expand":         return base + " " + EXPAND
    if condition == "paraphrase":     return paraphrases.get(row["cve_id"], base)
    return base  # original, reorder

def build_prompt(row, condition, paraphrases):
    order = ORDER_B if condition == "reorder" else ORDER_A
    kev = str(row["kev"]).lower() in ("true", "1", "yes")
    vals = {
        "Description": desc_for(row, condition, paraphrases),
        "CVSS base score": f"{row['cvss_score']} ({row.get('cvss_severity','')})".strip(),
        "EPSS exploitation probability": f"{float(row['epss']):.2f} (percentile {float(row['percentile']):.2f})",
        "Known exploited (CISA KEV)": "yes" if kev else "no",
        "Affected asset": row["asset"],
    }
    lines = "\n".join(f"{k}: {vals[k]}" for k in order)
    return (f"{POLICY}\nVulnerability information:\n{lines}\n\n"
            "Do not show chain-of-thought or intermediate reasoning. "
            "Put the verdict first.\n"
            "Respond on exactly two lines:\n"
            "VERDICT: <ACT|ATTEND|TRACK>\nREASON: <one short sentence>")

def call_model(model, system, user, temperature, tries=4):
    body = {"model": model, "temperature": temperature, "max_tokens": 80,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    data = json.dumps(body).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(OR_URL, data=data, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception:
            if t == tries - 1:
                return ""
            time.sleep(2 * (t + 1))

def parse_verdict(text):
    text = text or ""
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    m = re.search(r"VERDICT\s*[:\-]?\s*(ACT|ATTEND|TRACK)", text, re.I)
    if m:
        return m.group(1).upper(), True
    # Some chat templates return "TRACK Reason: ..." without the requested VERDICT
    # prefix. Accept only when the output starts with the verdict label.
    m_start = re.match(r"\s*(ACT|ATTEND|TRACK)\b", text, re.I)
    if m_start:
        return m_start.group(1).upper(), True
    # Only accept a bare verdict if the model gave a short final answer. This avoids
    # parsing labels mentioned inside reasoning text as the actual decision.
    short = text.strip()
    m2 = re.fullmatch(r"\s*(ACT|ATTEND|TRACK)(?:\s+(?:REASON\s*[:\-]?)?.*|[.:;-].*)?", short, re.I | re.S)
    return (m2.group(1).upper(), True) if m2 and len(short) <= 120 else ("INVALID", False)

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ---------- paraphrase generation (one-time) ----------
def make_paraphrases(rows, model):
    out = {}
    sysmsg = "Rewrite the vulnerability description in different words but identical technical meaning. Keep every fact. Output only the rewrite."
    for i, row in enumerate(rows):
        out[row["cve_id"]] = call_model(model, sysmsg, row["description"], 0.3) or row["description"]
        if (i + 1) % 25 == 0:
            print(f"  paraphrased {i+1}/{len(rows)}", flush=True)
    json.dump(out, open("paraphrases.json", "w"), ensure_ascii=False, indent=0)
    print(f"Wrote paraphrases.json ({len(out)} entries)")

# ---------- metrics ----------
def metrics(raw_path="results_raw.csv", gold_path="dataset.csv"):
    rows = load_rows(raw_path)
    if not rows:
        sys.exit(f"No result rows found in {raw_path}. Run a judge first, or delete an empty results file from a failed run.")
    gold = {r["cve_id"]: r["gold_label"] for r in load_rows(gold_path)} if os.path.exists(gold_path) else {}
    # group verdicts: by (model, cve, condition) -> list of verdicts (one per repeat)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["model"], r["cve_id"], r["condition"])].append(r["parsed_verdict"])

    def majority(vs):
        valid = [v for v in vs if v in VAL]
        if not valid:
            return "INVALID"
        return Counter(valid).most_common(1)[0][0]

    models = sorted({r["model"] for r in rows})
    cves   = sorted({r["cve_id"] for r in rows})

    cons_rows, flip_rows = [], []
    for model in models:
        # self-inconsistency on 'original'
        incons = invalid_o = acc_hit = acc_n = 0
        n_orig = 0
        orig_maj = {}
        for cve in cves:
            vs = grp.get((model, cve, "original"))
            if not vs:
                continue
            n_orig += 1
            valid = [v for v in vs if v in VAL]
            if not valid:
                invalid_o += 1
            elif len(set(valid)) > 1:
                incons += 1
            m = majority(vs)
            orig_maj[cve] = m
            if cve in gold and m in VAL:
                acc_n += 1
                acc_hit += int(m == gold[cve])
        cons_rows.append({"model": model, "n": n_orig,
                          "self_inconsistency_rate": round(incons / n_orig, 4) if n_orig else 0,
                          "invalid_rate_original": round(invalid_o / n_orig, 4) if n_orig else 0,
                          "accuracy_vs_gold": round(acc_hit / acc_n, 4) if acc_n else 0})
        # per-condition flips vs original-majority
        for cond in CONDITIONS[1:] + (["paraphrase"] if any(k[2] == "paraphrase" for k in grp) else []):
            n = flips = downgr = upgr = inval = 0
            for cve in cves:
                if cve not in orig_maj or orig_maj[cve] == "INVALID":
                    continue
                vs = grp.get((model, cve, cond))
                if not vs:
                    continue
                cm = majority(vs)
                n += 1
                if cm == "INVALID":
                    inval += 1; continue
                if cm != orig_maj[cve]:
                    flips += 1
                    if VAL[cm] < VAL[orig_maj[cve]] and orig_maj[cve] in ("ACT", "ATTEND"):
                        downgr += 1
                    elif VAL[cm] > VAL[orig_maj[cve]]:
                        upgr += 1
            flip_rows.append({"model": model, "condition": cond, "n": n,
                              "flip_rate": round(flips / n, 4) if n else 0,
                              "safety_downgrade_rate": round(downgr / n, 4) if n else 0,
                              "over_escalation_rate": round(upgr / n, 4) if n else 0,
                              "invalid_rate": round(inval / n, 4) if n else 0})

    with open("summary_consistency.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(cons_rows[0].keys())).writeheader() or None
    # rewrite properly (header + rows)
    with open("summary_consistency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cons_rows[0].keys())); w.writeheader(); w.writerows(cons_rows)
    with open("summary_flips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flip_rows[0].keys())); w.writeheader(); w.writerows(flip_rows)

    print("\n=== HEADLINE ===")
    for r in cons_rows:
        print(f"{r['model']:35s} self-inconsistency {r['self_inconsistency_rate']*100:4.1f}%  "
              f"acc-vs-gold {r['accuracy_vs_gold']*100:4.1f}%  invalid {r['invalid_rate_original']*100:.1f}%")
    print("--- flips by condition (flip / safety-downgrade) ---")
    for r in flip_rows:
        print(f"{r['model']:35s} {r['condition']:14s} flip {r['flip_rate']*100:4.1f}%  "
              f"downgrade {r['safety_downgrade_rate']*100:4.1f}%")
    print("\nWrote summary_consistency.csv and summary_flips.csv")

# ---------- main run ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--out", default="results_raw.csv")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="use only first N CVEs (0 = all)")
    ap.add_argument("--make-paraphrases", action="store_true")
    ap.add_argument("--para-model", default="openai/gpt-4o-mini")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    if args.metrics_only:
        metrics(args.out, args.dataset); return
    if not KEY:
        sys.exit("Set OPENROUTER_API_KEY first.")

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[:args.limit]

    if args.make_paraphrases:
        make_paraphrases(rows, args.para_model); return

    paraphrases = json.load(open("paraphrases.json")) if os.path.exists("paraphrases.json") else {}
    conditions = CONDITIONS + (["paraphrase"] if paraphrases else [])

    # resume support
    done = set()
    if os.path.exists(args.out):
        for r in load_rows(args.out):
            done.add((r["cve_id"], r["model"], r["condition"], r["repeat"]))
        print(f"Resuming: {len(done)} cells already done.")

    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    fieldnames = ["cve_id", "stratum", "asset", "gold_label", "model", "condition",
                  "repeat", "parsed_verdict", "parse_ok", "raw_output"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new:
        w.writeheader()

    total = len(rows) * len(MODELS) * len(conditions) * args.repeats
    n = 0
    for row in rows:
        prompts = {c: build_prompt(row, c, paraphrases) for c in conditions}
        for model in MODELS:
            for cond in conditions:
                for k in range(args.repeats):
                    n += 1
                    if (row["cve_id"], model, cond, str(k)) in done:
                        continue
                    out = call_model(model, SYSTEM, prompts[cond], args.temperature)
                    verdict, ok = parse_verdict(out)
                    w.writerow({"cve_id": row["cve_id"], "stratum": row["stratum"],
                                "asset": row["asset"], "gold_label": row["gold_label"],
                                "model": model, "condition": cond, "repeat": k,
                                "parsed_verdict": verdict, "parse_ok": ok,
                                "raw_output": (out or "").replace("\n", " ")[:2000]})
                    if n % 50 == 0:
                        f.flush(); print(f"  {n}/{total}", flush=True)
    f.close()
    print(f"Done -> {args.out}")
    metrics(args.out, args.dataset)

if __name__ == "__main__":
    main()
