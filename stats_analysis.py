#!/usr/bin/env python3
"""Statistical summaries for the full LLM triage-stability experiment.

Inputs:
  - results_full.csv
  - dataset.csv (only used to preserve/validate CVE universe if present)

Outputs:
  - stats_rates_ci.csv
  - stats_reorder_tests.csv
  - paper_table_rates.md
"""
import argparse
import csv
import math
from collections import Counter, defaultdict

VAL = {"ACT": 2, "ATTEND": 1, "TRACK": 0}
CONDITION_ORDER = ["reorder", "add_lowstakes", "add_highstakes", "compress", "expand", "paraphrase"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def majority(verdicts):
    valid = [v for v in verdicts if v in VAL]
    if not valid:
        return "INVALID"
    return Counter(valid).most_common(1)[0][0]


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def binom_sf(k, n, p=0.5):
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def mcnemar_exact_p(b, c):
    discordant = b + c
    if discordant == 0:
        return 1.0
    lo = min(b, c)
    hi = max(b, c)
    return min(1.0, 2 * min(binom_cdf(lo, discordant), binom_sf(hi, discordant)))


def holm_adjust(pairs):
    """Return Holm-adjusted p-values for [(key, p), ...]."""
    m = len(pairs)
    ordered = sorted(enumerate(pairs), key=lambda x: x[1][1])
    adjusted = [None] * m
    running = 0.0
    for rank, (orig_i, (_key, p)) in enumerate(ordered):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted[orig_i] = running
    return adjusted


def pct(x):
    return f"{100 * x:.1f}%"


def pct_ci(lo, hi):
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_full.csv")
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--rates-out", default="stats_rates_ci.csv")
    ap.add_argument("--tests-out", default="stats_reorder_tests.csv")
    ap.add_argument("--table-out", default="paper_table_rates.md")
    args = ap.parse_args()

    rows = read_csv(args.results)

    grp = defaultdict(list)
    for r in rows:
        grp[(r["model"], r["cve_id"], r["condition"])].append(r["parsed_verdict"])

    models = sorted({r["model"] for r in rows})
    cves = sorted({r["cve_id"] for r in rows})
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]

    maj = {key: majority(vs) for key, vs in grp.items()}

    rates = []
    flip_indicator = defaultdict(dict)
    for model in models:
        for cond in conditions:
            n = flip_count = down_count = up_count = 0
            for cve in cves:
                orig = maj.get((model, cve, "original"), "INVALID")
                cur = maj.get((model, cve, cond), "INVALID")
                if orig not in VAL or cur not in VAL:
                    continue
                flipped = cur != orig
                flip_indicator[(model, cond)][cve] = int(flipped)
                n += 1
                if flipped:
                    flip_count += 1
                    if VAL[cur] < VAL[orig] and orig in ("ACT", "ATTEND"):
                        down_count += 1
                    elif VAL[cur] > VAL[orig]:
                        up_count += 1

            f_lo, f_hi = wilson_ci(flip_count, n)
            d_lo, d_hi = wilson_ci(down_count, n)
            u_lo, u_hi = wilson_ci(up_count, n)
            rates.append({
                "model": model,
                "condition": cond,
                "n": n,
                "flip_count": flip_count,
                "flip_rate": round(flip_count / n, 6) if n else 0,
                "flip_ci_low": round(f_lo, 6),
                "flip_ci_high": round(f_hi, 6),
                "safety_downgrade_count": down_count,
                "safety_downgrade_rate": round(down_count / n, 6) if n else 0,
                "safety_downgrade_ci_low": round(d_lo, 6),
                "safety_downgrade_ci_high": round(d_hi, 6),
                "over_escalation_count": up_count,
                "over_escalation_rate": round(up_count / n, 6) if n else 0,
                "over_escalation_ci_low": round(u_lo, 6),
                "over_escalation_ci_high": round(u_hi, 6),
            })

    with open(args.rates_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rates[0].keys()))
        w.writeheader()
        w.writerows(rates)

    tests = []
    for model in models:
        raw_tests = []
        for cond in conditions:
            if cond == "reorder":
                continue
            b = c = both_n = 0
            for cve in cves:
                r = flip_indicator.get((model, "reorder"), {}).get(cve)
                o = flip_indicator.get((model, cond), {}).get(cve)
                if r is None or o is None:
                    continue
                both_n += 1
                if r == 1 and o == 0:
                    b += 1
                elif r == 0 and o == 1:
                    c += 1
            p = mcnemar_exact_p(b, c)
            raw_tests.append((cond, p, b, c, both_n))

        adjusted = holm_adjust([((model, cond), p) for cond, p, _b, _c, _n in raw_tests])
        for (cond, p, b, c, both_n), p_holm in zip(raw_tests, adjusted):
            tests.append({
                "model": model,
                "comparison": f"reorder_vs_{cond}",
                "n_paired": both_n,
                "reorder_only_flips_b": b,
                "other_only_flips_c": c,
                "mcnemar_exact_p": round(p, 8),
                "holm_p": round(p_holm, 8),
                "significant_0_05": p_holm < 0.05,
            })

    with open(args.tests_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        w.writeheader()
        w.writerows(tests)

    lines = [
        "| Model | Condition | n | Flip % (95% CI) | Downgrade % (95% CI) | Over-escalation % (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rates:
        model = r["model"].split("/")[-1]
        lines.append(
            f"| {model} | {r['condition']} | {r['n']} | "
            f"{pct(float(r['flip_rate']))} {pct_ci(float(r['flip_ci_low']), float(r['flip_ci_high']))} | "
            f"{pct(float(r['safety_downgrade_rate']))} {pct_ci(float(r['safety_downgrade_ci_low']), float(r['safety_downgrade_ci_high']))} | "
            f"{pct(float(r['over_escalation_rate']))} {pct_ci(float(r['over_escalation_ci_low']), float(r['over_escalation_ci_high']))} |"
        )
    with open(args.table_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.rates_out}")
    print(f"Wrote {args.tests_out}")
    print(f"Wrote {args.table_out}")


if __name__ == "__main__":
    main()
