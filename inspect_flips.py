#!/usr/bin/env python3
"""
Pull real before/after examples where a meaning-preserving rewrite flipped the
verdict - ready-made qualitative cases for the Discussion section.

  python inspect_flips.py --top 12                 # all flips, downgrades first
  python inspect_flips.py --top 12 --downgrades-only

Writes flips_examples.txt and prints to screen.
"""
import argparse, csv, json, os
from collections import defaultdict, Counter
from run_and_score import VAL, load_rows, desc_for

def majority(vs):
    valid = [v for v in vs if v in VAL]
    return Counter(valid).most_common(1)[0][0] if valid else "INVALID"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results_raw.csv")
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--downgrades-only", action="store_true")
    args = ap.parse_args()

    meta = {r["cve_id"]: r for r in load_rows(args.dataset)}
    paraphrases = json.load(open("paraphrases.json")) if os.path.exists("paraphrases.json") else {}
    grp = defaultdict(list)
    for r in load_rows(args.raw):
        grp[(r["model"], r["cve_id"], r["condition"])].append(r["parsed_verdict"])

    examples = []
    for (model, cve, cond), vs in grp.items():
        if cond == "original":
            continue
        orig = grp.get((model, cve, "original"))
        if not orig:
            continue
        om, cm = majority(orig), majority(vs)
        if om in VAL and cm in VAL and cm != om:
            down = VAL[cm] < VAL[om] and om in ("ACT", "ATTEND")
            if args.downgrades_only and not down:
                continue
            examples.append((down, model, cve, cond, om, cm))

    examples.sort(key=lambda e: not e[0])  # downgrades first
    lines = []
    for down, model, cve, cond, om, cm in examples[:args.top]:
        m = meta.get(cve, {})
        base = m.get("description", "")
        changed = desc_for({**m}, cond, paraphrases)
        tag = "DOWNGRADE" if down else "change"
        note = "(order changed; description identical)" if cond == "reorder" else ""
        lines.append(
            f"[{tag}] {model.split('/')[-1]} | {cve} | condition={cond} | {om} -> {cm}\n"
            f"  facts: CVSS {m.get('cvss_score')}  EPSS {m.get('epss')}  KEV {m.get('kev')}  asset={m.get('asset')}\n"
            f"  original : {base[:200]}\n"
            f"  perturbed: {changed[:200]} {note}\n")
    out = "\n".join(lines) if lines else "No verdict flips found."
    open("flips_examples.txt", "w", encoding="utf-8").write(out)
    print(out)
    print(f"\n({len(examples)} flips total; showing up to {args.top}) -> flips_examples.txt")

if __name__ == "__main__":
    main()
