#!/usr/bin/env python3
"""
STEP 1 - Build the base vulnerability dataset for the LLM-triage stability study.

Joins three PUBLIC signals per CVE:
  - description + CVSS base score   -> NVD CVE API 2.0
  - EPSS exploitation probability   -> FIRST EPSS bulk CSV
  - KEV "known exploited" flag      -> CISA KEV JSON

Then assigns a fixed asset context, computes a gold ACT/ATTEND/TRACK label from a
transparent policy, tags a conflict stratum, and writes dataset.csv.

Usage:
  # smoke test first (tiny, ~1 min) to confirm the joins/parsing work:
  python pull_data.py --target-per-stratum 5 --out dataset_smoke.csv
  # then the real pull:
  python pull_data.py --target-per-stratum 80 --out dataset.csv
Optional (faster, higher NVD rate limit):  export NVD_API_KEY=xxxxxxxx

NOTE: not tested against the live APIs in this environment - run the smoke test
and eyeball dataset_smoke.csv before the full pull.
"""
import argparse, csv, gzip, json, os, random, sys, time, urllib.request
from collections import defaultdict

NVD_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
EPSS_CSV = "https://epss.cyentia.com/epss_scores-current.csv.gz"
KEV_JSON = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# (asset description, is_critical_or_internet_facing)
ASSETS = [
    ("internet-facing production web server", True),
    ("internal developer workstation",        False),
    ("hospital patient-records server",        True),
    ("isolated lab test machine",              False),
]
STRATA = ["G1_kev", "G2_highsev_lowprob", "G3_highprob_notkev", "G4_low"]

def fetch(url, headers=None, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))

def load_kev():
    data = json.loads(fetch(KEV_JSON))
    return {v["cveID"] for v in data.get("vulnerabilities", [])}

def load_epss():
    text = gzip.decompress(fetch(EPSS_CSV)).decode("utf-8")
    epss = {}
    for line in text.splitlines():
        if not line or line[0] == "#" or line.startswith("cve"):
            continue
        p = line.split(",")
        if len(p) >= 3:
            try:
                epss[p[0]] = (float(p[1]), float(p[2]))  # (epss, percentile)
            except ValueError:
                pass
    return epss

def nvd_lookup(cve_id, api_key, sleep):
    headers = {"apiKey": api_key} if api_key else {}
    time.sleep(sleep)
    try:
        data = json.loads(fetch(f"{NVD_URL}?cveId={cve_id}", headers))
    except Exception:
        return None
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None
    cve = vulns[0]["cve"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
    score = severity = vector = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if cve.get("metrics", {}).get(key):
            m = cve["metrics"][key][0]
            cd = m.get("cvssData", {})
            score = cd.get("baseScore")
            severity = cd.get("baseSeverity") or m.get("baseSeverity") or ""
            vector = cd.get("vectorString") or ""
            break
    if not desc or score is None:
        return None
    desc = " ".join(desc.split())  # collapse whitespace/newlines
    return {"description": desc, "cvss_score": float(score),
            "cvss_severity": severity, "cvss_vector": vector}

def gold_label(cvss, epss, kev, critical):
    if kev or (epss >= 0.5 and critical) or (cvss >= 9.0 and epss >= 0.3):
        return "ACT"
    if (cvss >= 7.0 and epss >= 0.1) or epss >= 0.3 or (critical and cvss >= 7.0):
        return "ATTEND"
    return "TRACK"

def stratum_of(cvss, epss, kev):
    if kev:                              return "G1_kev"
    if cvss >= 7.0 and epss < 0.1:       return "G2_highsev_lowprob"
    if epss >= 0.3:                      return "G3_highprob_notkev"
    return "G4_low"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-stratum", type=int, default=80)
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-calls", type=int, default=6000)
    args = ap.parse_args()
    random.seed(args.seed)
    api_key = os.environ.get("NVD_API_KEY")
    sleep = 0.4 if api_key else 0.8  # respect NVD rate limits

    print("Loading KEV ...", flush=True); kev = load_kev()
    print(f"  KEV entries: {len(kev)}")
    print("Loading EPSS ...", flush=True); epss = load_epss()
    print(f"  EPSS rows: {len(epss)}")

    # Candidate pools, built from KEV/EPSS knowledge (no NVD call needed yet)
    kev_pool     = [c for c in epss if c in kev]
    g3_pool      = [c for c, (e, _) in epss.items() if e >= 0.3 and c not in kev]
    lowprob_pool = [c for c, (e, _) in epss.items() if e < 0.1 and c not in kev]
    for pool in (kev_pool, g3_pool, lowprob_pool):
        random.shuffle(pool)

    rows, counts, calls = [], defaultdict(int), 0
    tgt = args.target_per_stratum

    def accept(cve_id):
        nonlocal calls
        if calls >= args.max_calls:
            return None
        calls += 1
        e, pct = epss[cve_id]
        info = nvd_lookup(cve_id, api_key, sleep)
        if not info:
            return None
        asset, crit = random.choice(ASSETS)
        s = stratum_of(info["cvss_score"], e, kev_member)
        gl = gold_label(info["cvss_score"], e, kev_member, crit)
        return {"cve_id": cve_id, "stratum": s, "asset": asset, "critical": crit,
                "epss": round(e, 4), "percentile": round(pct, 4),
                "kev": kev_member, "gold_label": gl, **info}

    # Fill G1 (KEV), G3 (high prob), then sweep low-prob to fill G2 + G4
    for pool, want in ((kev_pool, "G1_kev"), (g3_pool, "G3_highprob_notkev")):
        for cve_id in pool:
            if counts[want] >= tgt or calls >= args.max_calls:
                break
            kev_member = cve_id in kev
            r = accept(cve_id)
            if r and r["stratum"] == want:
                rows.append(r); counts[want] += 1
                if len(rows) % 20 == 0:
                    print(f"  collected {len(rows)} (calls={calls})", flush=True)

    for cve_id in lowprob_pool:
        if (counts["G2_highsev_lowprob"] >= tgt and counts["G4_low"] >= tgt) or calls >= args.max_calls:
            break
        kev_member = False
        r = accept(cve_id)
        if not r:
            continue
        s = r["stratum"]
        if s in ("G2_highsev_lowprob", "G4_low") and counts[s] < tgt:
            rows.append(r); counts[s] += 1
            if len(rows) % 20 == 0:
                print(f"  collected {len(rows)} (calls={calls})", flush=True)

    fields = ["cve_id", "stratum", "asset", "critical", "cvss_score", "cvss_severity",
              "cvss_vector", "epss", "percentile", "kev", "gold_label", "description"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows -> {args.out}  (NVD calls: {calls})")
    print("Per stratum:", dict(counts))
    gd = defaultdict(int)
    for r in rows:
        gd[r["gold_label"]] += 1
    print("Gold label distribution:", dict(gd))
    print("If a stratum is far below target, raise --max-calls or lower --target-per-stratum.")

if __name__ == "__main__":
    main()
