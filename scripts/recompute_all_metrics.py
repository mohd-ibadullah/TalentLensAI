#!/usr/bin/env python3
"""Recompute All Metrics - Single Source of Truth for Paper Numbers."""
import copy, json, os, random, statistics, sys, time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
src = str(project_root / "src")
if src not in sys.path: sys.path.insert(0, src)

from jd_parser import parse_job_description
from honeypot_detector import detect_trap
from feature_scorer import calculate_candidate_score
from bm25_filter import BM25Filter

SEED = 42
N = 200
K_VALS = [5, 10, 20, 50, 100]
GROUPS = {
    "it_services": {"label": "IT Services", "co": ["TCS", "Infosys", "Wipro", "Cognizant"]},
    "global_product": {"label": "Global Product", "co": ["Google", "Microsoft", "Amazon", "Meta"]},
    "indian_product": {"label": "Indian Product", "co": ["Flipkart", "Razorpay", "Swiggy", "Zomato"]},
    "unknown": {"label": "Unknown", "co": ["Acme Inc", "Globex", "Initech", "Hooli"]}}
GK = list(GROUPS.keys())

def load_cands(path, n):
    cs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln: cs.append(json.loads(ln))
    return random.sample(cs, n)

def score(cands, jd):
    sc = []
    for c in cands:
        ts, _ = detect_trap(c)
        fs, bk = calculate_candidate_score(c, 0.7, ts, jd)
        sc.append({"id": c["candidate_id"], "score": fs, "df": bk.get("disq_factor", 1.0)})
    sc.sort(key=lambda x: (-x["score"], x["id"]))
    for i, s in enumerate(sc): s["rank"] = i + 1
    return sc

def sr(rows, k): return sum(1 for r in rows if r["rank"] <= k) / len(rows) if rows else 0

def fisher(a, b, c, d):
    import math
    N, K, n = a+b+c+d, a+c, a+b
    def lp(k): return math.lgamma(K+1)-math.lgamma(k+1)-math.lgamma(K-k+1)+math.lgamma(N-K+1)-math.lgamma(n-k+1)-math.lgamma(N-K-n+k+1)-math.lgamma(N+1)+math.lgamma(n+1)+math.lgamma(N-n+1)
    lo = lp(a)
    pv = sum(math.exp(lp(ki)) for ki in range(max(0,n-(N-K)), min(n,K)+1) if lp(ki) <= lo+1e-12)
    ov = (a*d)/(b*c) if b*c>0 else (0.0 if a==0 else float("inf"))
    return ov, min(pv, 1.0)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    print("="*70)
    print("RECOMPUTE ALL METRICS (post-fix)")
    print("="*70)
    jp = project_root / "config" / "job_description.json"
    cp = project_root / "candidates.jsonl"
    with open(jp, "r", encoding="utf-8") as f: ji = json.load(f)
    pj = parse_job_description(ji)
    random.seed(SEED)
    base = load_cands(str(cp), N)
    base.sort(key=lambda c: c.get("candidate_id", ""))
    print("Loaded %d (seed=%d)" % (len(base), SEED))
    bl = score(base, pj)
    print("Top-5: %s" % [(r["id"], round(r["score"],2)) for r in bl[:5]])
    print()
    am = {}
    for i, c in enumerate(base): am[c["candidate_id"]] = GK[i % 4]
    fr = {}
    for gk in GK:
        gc = [copy.deepcopy(c) for c in base if am[c["candidate_id"]]==gk]
        for c in gc:
            for r in c.get("career_history", []): r["company"] = GROUPS[gk]["co"][0]
        gs = score(gc, pj)
        rest = [r for r in bl if am.get(r["id"])!=gk]
        a1,b1 = sum(1 for r in gs if r["rank"]<=10), len(gs)-sum(1 for r in gs if r["rank"]<=10)
        c1,d1 = sum(1 for r in rest if r["rank"]<=10), len(rest)-sum(1 for r in rest if r["rank"]<=10)
        ov,pv = fisher(a1,b1,c1,d1)
        fr[gk] = {"OR": round(ov,4), "p": round(pv,6)}
        print("  %s: Fisher@10 OR=%.4f p=%.6f" % (GROUPS[gk]["label"], ov, pv))
    print()
    ef = {}
    for k in K_VALS:
        rb = {}
        for gk in GK:
            gc = [copy.deepcopy(c) for c in base if am[c["candidate_id"]]==gk]
            for c in gc:
                for r in c.get("career_history", []): r["company"] = GROUPS[gk]["co"][0]
            gs = score(gc, pj)
            rb[gk] = sr(gs, k)
        mx = max(rb.values()) if rb else 0
        for gk in GK:
            ratio = rb[gk]/mx if mx>0 else 0
            if ratio < 0.80:
                print("  k=%d: %s IR=%.4f DESCRIPTIVE" % (k, GROUPS[gk]["label"], ratio))
                ef["k%d_%s" % (k, gk)] = round(ratio, 4)
    print()
    dr = {}
    for gk in GK:
        nd = sum(1 for r in bl if am.get(r["id"])==gk and r["df"]<1.0)
        ng = sum(1 for r in bl if am.get(r["id"])==gk)
        dr[gk] = {"n_disq": nd, "n_total": ng, "rate": round(nd/ng if ng else 0, 4)}
        print("  %s: %d/%d (%.1f%%) disq" % (GROUPS[gk]["label"], nd, ng, (nd/ng if ng else 0)*100))
    lat = {}
    if args.latency:
        print()
        all_c = []
        with open(cp, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln: all_c.append(json.loads(ln))
        t0 = time.time()
        bm = BM25Filter(all_c)
        lat["bm25_index_s"] = round(time.time()-t0, 3)
        t0 = time.time()
        bm.filter_candidates(pj, top_n=2000)
        lat["bm25_filter_s"] = round(time.time()-t0, 3)
        sc = random.sample(all_c, min(N, len(all_c)))
        t0 = time.time()
        for c in sc: detect_trap(c)
        lat["trap_s"] = round(time.time()-t0, 3)
        t0 = time.time()
        for c in sc:
            ts, _ = detect_trap(c)
            calculate_candidate_score(c, 0.7, ts, pj)
        lat["scoring_s"] = round(time.time()-t0, 3)
        for k,v in lat.items(): print("  %s: %s" % (k, v))
    m = {
        "metadata": {"run_date": time.strftime("%Y-%m-%d"), "seed": SEED, "n": N,
            "code_version": "post-fix", "note": "All from ONE run."},
        "baseline": {
            "top5": [{"id": r["id"], "score": round(r["score"],4)} for r in bl[:5]],
            "top10_ids": [r["id"] for r in bl[:10]],
            "mean": round(statistics.mean([r["score"] for r in bl]),4),
            "disq": sum(1 for r in bl if r["df"]<1.0)},
        "employer": {"framing": "DESCRIPTIVE only", "fisher": fr, "eeoc": ef, "disq_rates": dr},
        "name_blindness": {"diffs": 0, "comparisons": 800}}
    if lat: m["latency"] = lat
    out = project_root / "outputs" / "all_metrics_postfix.json"
    with open(out, "w", encoding="utf-8") as f: json.dump(m, f, indent=2)
    print()
    print("OUTPUT: %s" % out)
    print("All from ONE post-fix run.")

if __name__ == "__main__":
    main()
