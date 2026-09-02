#!/usr/bin/env python3
"""
Employer round-robin impact measurement — v2 (multi-k, Fisher exact, honest).

Assigns employers round-robin across 4 groups so all groups coexist in ONE
ranking. Measures selection-rate differences by group and computes EEOC-style
impact ratios (group rate / highest rate) at MULTIPLE k values (5, 10, 20,
50, 100). Includes Fisher's exact test for IT Services vs rest at each k.

Tables produced:
  Table 1 - Multi-k selection rates & impact ratio (treatment)
  Table 2 - Score distribution per group (treatment)
  Table 3 - Rank distribution per group (treatment)
  Top-20 ranking verbatim
  Fisher exact test: IT Services vs rest at each k
  Sanity check: disqualifier firing asymmetry (honest 50/50 vs 4/50)
  Score ceiling analysis: IT Services max vs other groups' means
  Consistency check: re-derive aggregates from per-candidate table
  Control run: same analysis on untouched employers

IMPORTANT FRAMING NOTE (v2):
  The EEOC four-fifths impact ratios (IR) below are DESCRIPTIVE FLAGS only.
  With n=50 per group (total 200), this test is underpowered for causal
  adverse-impact claims. Fisher exact test (shown separately) is the
  appropriate significance test at this sample size. Do NOT claim
  'adverse impact' from the IR columns alone.
"""

import copy
import json
import math
import random
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from jd_parser import parse_job_description
from honeypot_detector import detect_trap
from feature_scorer import calculate_candidate_score

# ── Configuration ─────────────────────────────────────────────────────────────

N_CANDIDATES = 200
JD_PATH = project_root / "config" / "job_description.json"
CANDIDATES_PATH = project_root / "candidates.jsonl"

GROUPS = {
    "it_services": {"label": "IT Services", "companies": ["TCS", "Infosys", "Wipro", "Cognizant"]},
    "global_product": {"label": "Global Product", "companies": ["Google", "Microsoft", "Amazon", "Meta"]},
    "indian_product": {"label": "Indian Product", "companies": ["Flipkart", "Razorpay", "Swiggy", "Zomato"]},
    "unknown": {"label": "Unknown", "companies": ["Acme Inc", "Globex", "Initech", "Hooli"]},
}

GROUP_KEYS = ["it_services", "global_product", "indian_product", "unknown"]
K_VALUES = [5, 10, 20, 50, 100]
SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_candidates(path: str, n: int) -> list[dict]:
    """Load ALL candidates, then sample n at random with SEED for reproducibility."""
    all_cands = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_cands.append(json.loads(line))
    print(f"  Total candidates on disk: {len(all_cands):,}")
    sampled = random.sample(all_cands, n)
    return sampled


def classify_title(cand: dict) -> str:
    """Classify a candidate into tech / non-tech / HR-Marketing by title."""
    title = (cand.get("profile", {}).get("current_title", "") or "").lower()
    hr_mktg = ["marketing", "hr ", "hr manager", "human resources",
               "recruiter", "sales", "talent"]
    non_tech = ["mechanical engineer", "civil engineer", "electrical engineer",
                "chemical engineer", "structural engineer", "environmental engineer",
                "industrial engineer"]
    if any(kw in title for kw in hr_mktg):
        return "HR-Marketing"
    if any(kw in title for kw in non_tech):
        return "Non-Tech"
    return "Tech"


def report_sample_composition(candidates: list[dict]) -> None:
    """Print how many candidates by title category in the sample."""
    cats = {"Tech": 0, "Non-Tech": 0, "HR-Marketing": 0}
    for c in candidates:
        cats[classify_title(c)] += 1
    print(f"\nSample composition ({len(candidates)} candidates, seed={SEED}):")
    for cat, count in cats.items():
        print(f"  {cat:<15} {count:>5}  ({count/len(candidates)*100:.1f}%)")


def swap_companies(candidates: list[dict], company: str) -> list[dict]:
    result = []
    for cand in candidates:
        c = copy.deepcopy(cand)
        for role in c.get("career_history", []):
            role["company"] = company
        result.append(c)
    return result


def score_candidates(candidates: list[dict], parsed_jd: dict) -> list[dict]:
    """Run trap detection + feature scoring. Return sorted scored list."""
    scored = []
    for cand in candidates:
        trap_score, _ = detect_trap(cand)
        final_score, breakdown = calculate_candidate_score(
            cand, 0.7, trap_score, parsed_jd
        )
        scored.append({
            "candidate_id": cand["candidate_id"],
            "score": final_score,
            "breakdown": breakdown,
            "trap_score": trap_score,
            "disq_penalty": breakdown.get("disqualifier_penalty_applied", 0.0),
            "disq_factor": breakdown.get("disq_factor", 1.0),
        })
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, s in enumerate(scored):
        s["rank"] = i + 1
    return scored


def build_delta_table(
    baseline: list[dict], swapped: list[dict]
) -> list[dict]:
    b_map = {r["candidate_id"]: r for r in baseline}
    s_map = {r["candidate_id"]: r for r in swapped}
    deltas = []
    for cid in b_map:
        if cid not in s_map:
            continue
        b = b_map[cid]
        s = s_map[cid]
        score_delta = s["score"] - b["score"]
        rank_delta = s["rank"] - b["rank"]
        deltas.append({
            "candidate_id": cid,
            "baseline_score": b["score"],
            "swapped_score": s["score"],
            "score_delta": score_delta,
            "abs_score_delta": abs(score_delta),
            "baseline_rank": b["rank"],
            "swapped_rank": s["rank"],
            "rank_delta": rank_delta,
            "abs_rank_delta": abs(rank_delta),
            "baseline_disq": b["disq_penalty"],
            "swapped_disq": s["disq_penalty"],
            "baseline_disq_factor": b["disq_factor"],
            "swapped_disq_factor": s["disq_factor"],
        })
    return deltas


def derive_metrics(deltas: list[dict], baseline_top50: set, swapped_top50: set,
                   baseline_top10: list, swapped_top10: list) -> dict:
    abs_deltas = [d["abs_score_delta"] for d in deltas]
    abs_ranks = [d["abs_rank_delta"] for d in deltas]
    abs_deltas_sorted = sorted(abs_deltas)
    abs_ranks_sorted = sorted(abs_ranks)

    def p95(arr):
        if not arr:
            return 0.0
        idx = min(int(len(arr) * 0.95), len(arr) - 1)
        return arr[idx]

    baseline_top50_ids = set(baseline_top50)
    swapped_top50_ids = set(swapped_top50)
    top50_entries = swapped_top50_ids - baseline_top50_ids
    top50_exits = baseline_top50_ids - swapped_top50_ids
    top50_turnover = len(top50_entries) + len(top50_exits)

    baseline_top10_ids = set(baseline_top10)
    swapped_top10_ids = set(swapped_top10)
    top10_added = swapped_top10_ids - baseline_top10_ids
    top10_removed = baseline_top10_ids - swapped_top10_ids

    return {
        "score_delta_max": max(abs_deltas) if abs_deltas else 0.0,
        "score_delta_p95": p95(abs_deltas_sorted),
        "top50_entries": top50_entries,
        "top50_exits": top50_exits,
        "top50_turnover": top50_turnover,
        "rank_delta_max": max(abs_ranks) if abs_ranks else 0,
        "rank_delta_p95": p95(abs_ranks_sorted),
        "top10_before": list(baseline_top10_ids),
        "top10_after": list(swapped_top10_ids),
        "top10_added": list(top10_added),
        "top10_removed": list(top10_removed),
        "top10_changed": bool(top10_added or top10_removed),
    }


def assign_employers_round_robin(candidates):
    assignment_map = {}
    result = []
    for i, cand in enumerate(candidates):
        c = copy.deepcopy(cand)
        gk = GROUP_KEYS[i % 4]
        ci = (i // 4) % 4
        co = GROUPS[gk]["companies"][ci]
        assignment_map[c["candidate_id"]] = {"group": gk, "gl": GROUPS[gk]["label"], "company": co}
        for role in c.get("career_history", []):
            role["company"] = co
        result.append(c)
    return result, assignment_map

def build_table(scored, assignment_map):
    return [{"candidate_id": s["candidate_id"], "group": assignment_map.get(s["candidate_id"], {}).get("group", "?"),
             "group_label": assignment_map.get(s["candidate_id"], {}).get("gl", "?"),
             "company": assignment_map.get(s["candidate_id"], {}).get("company", "?"),
             "score": s["score"], "rank": s["rank"], "disq_factor": s["disq_factor"], "disq_penalty": s["disq_penalty"]}
            for s in scored]

# ── Multi-k selection rate ────────────────────────────────────────────────────

def sel_rate_multik(rows, k_values=K_VALUES):
    """Compute selection rate at multiple k values."""
    n = len(rows)
    result = {"n": n}
    for k in k_values:
        count = sum(1 for r in rows if r["rank"] <= k)
        result[f"in_top_{k}"] = count
        result[f"rate_{k}"] = count / n if n else 0
    return result


def sd(rows):
    sc = [r["score"] for r in rows]
    dq = sum(1 for r in rows if r["disq_factor"] < 1.0)
    if not sc:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "dq": 0, "scores": []}
    return {
        "mean": statistics.mean(sc), "median": statistics.median(sc),
        "min": min(sc), "max": max(sc), "dq": dq, "scores": sc,
    }

def rd(rows):
    rk = [r["rank"] for r in rows]
    t10 = sum(1 for r in rows if r["rank"] <= 10)
    if not rk:
        return {"mean": 0, "median": 0, "best": 0, "worst": 0, "t10": 0}
    return {"mean": statistics.mean(rk), "median": statistics.median(rk),
            "best": min(rk), "worst": max(rk), "t10": t10}

# ── Fisher's exact test (2x2 table, hypergeometric, no scipy) ────────────────

def _log_pmf_hypergeom(N, K, n, k):
    """Log probability of hypergeometric: P(X=k) = C(K,k)*C(N-K,n-k)/C(N,n).

    Uses math.lgamma (Python built-in, no scipy needed).
    """
    return (math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
            + math.lgamma(N - K + 1) - math.lgamma(n - k + 1) - math.lgamma(N - K - n + k + 1)
            - math.lgamma(N + 1) + math.lgamma(n + 1) + math.lgamma(N - n + 1))


def fisher_exact_2x2(a, b, c, d):
    """
    Fisher's exact test for a 2x2 contingency table:
        |  in_top_k  |  not_in_top_k  |
        |------------|----------------|
    grp |     a      |       b        |
    rst |     c      |       d        |

    Returns (odds_ratio, p_value) where p_value is two-tailed.
    """
    N = a + b + c + d
    K = a + c  # total "in top k"
    n = a + b  # total for group of interest
    k = a      # observed successes

    # Compute p-value: sum of all tables at least as extreme
    log_p_obs = _log_pmf_hypergeom(N, K, n, k)

    # Two-tailed: sum probabilities with P(X) <= P(X_obs)
    p_value = 0.0
    k_min = max(0, n - (N - K))
    k_max = min(n, K)

    for ki in range(k_min, k_max + 1):
        log_p = _log_pmf_hypergeom(N, K, n, ki)
        if log_p <= log_p_obs + 1e-12:  # include ties
            p_value += math.exp(log_p)

    # Odds ratio
    if b * c > 0:
        odds_ratio = (a * d) / (b * c)
    elif a == 0:
        odds_ratio = 0.0
    else:
        odds_ratio = float('inf')

    return odds_ratio, min(p_value, 1.0)

# ── Print helpers ─────────────────────────────────────────────────────────────

def pt1_multi_k(s):
    """Table 1 — multi-k selection rates & impact ratio."""
    print("\nTable 1 - Selection rates & impact ratio (multi-k)")
    # Header
    hdr_parts = [f"{'group':<20} {'n':>4}"]
    for k in K_VALUES:
        hdr_parts.append(f"top-{k:>3}")
    for k in K_VALUES:
        hdr_parts.append(f"rate@{k:>3}")
    for k in K_VALUES:
        hdr_parts.append(f"IR@{k:>3}")
    print("  ".join(hdr_parts))
    print("-" * (20 + 4 + 7 * len(K_VALUES) + 7 * len(K_VALUES) + 7 * len(K_VALUES) + 3 * len(K_VALUES) + 5 * len(K_VALUES)))

    # Compute max rates per k for impact ratio
    max_rates = {}
    for k in K_VALUES:
        max_rates[k] = max(s[gk][f"rate_{k}"] for gk in GROUP_KEYS)

    for gk in GROUP_KEYS:
        r = s[gk]
        parts = [f"{GROUPS[gk]['label']:<20} {r['n']:>4}"]
        for k in K_VALUES:
            parts.append(f"{r[f'in_top_{k}']:>7}")
        for k in K_VALUES:
            parts.append(f"{r[f'rate_{k}']:>7.4f}")
        for k in K_VALUES:
            mx = max_rates[k]
            ratio = r[f"rate_{k}"] / mx if mx else 0
            marker = " ***" if ratio < 0.80 else ""  # descriptive flag only (underpowered at n=50)
            parts.append(f"{ratio:>7.4f}{marker}")
        print("  ".join(parts))

    # Disclaimer: four-fifths rule is descriptive only at this sample size
    print()
    print("  NOTE: *** = below EEOC 0.80 four-fifths threshold (DESCRIPTIVE FLAG ONLY).")
    print("  With n=50 per group, the impact ratio is underpowered for causal adverse-impact claims.")
    print("  Use Fisher exact test (below) for statistical significance at this sample size.")


def pt2(d):
    print("\nTable 2 - Score distribution per group")
    print(f"{'group':<20} {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'disq fires':>10}")
    print("-" * 70)
    for gk in GROUP_KEYS:
        r = d[gk]
        print(f"{GROUPS[gk]['label']:<20} {r['mean']:>8.4f} {r['median']:>8.4f} {r['min']:>8.4f} {r['max']:>8.4f} {r['dq']:>10}")


def pt3(d):
    print("\nTable 3 - Rank distribution per group")
    print(f"{'group':<20} {'mean rank':>9} {'median':>8} {'best':>5} {'worst':>6} {'in top-10':>10}")
    print("-" * 70)
    for gk in GROUP_KEYS:
        r = d[gk]
        print(f"{GROUPS[gk]['label']:<20} {r['mean']:>9.1f} {r['median']:>8.1f} {r['best']:>5} {r['worst']:>6} {r['t10']:>10}")

def pt20(table):
    print("\nTop-20 ranking (verbatim)")
    print(f"{'candidate_id':<16} {'group':<20} {'company':<14} {'score':>8} {'rank':>5}")
    print("-" * 70)
    for r in table[:20]:
        print(f"{r['candidate_id']:<16} {r['group_label']:<20} {r['company']:<14} {r['score']:>8.4f} {r['rank']:>5}")


def _fisher_self_test():
    """Self-test: balanced table (25,25,75,75) must give p ≈ 1.0."""
    or_val, p_val = fisher_exact_2x2(25, 25, 75, 75)
    assert abs(p_val - 1.0) < 0.01, f"Fisher self-test FAILED: balanced table gave p={p_val:.6f}, expected ≈ 1.0"
    # Edge case: all zeros (0,50,100,50) — IT gets 0, rest gets 100 of 150
    or2, p2 = fisher_exact_2x2(0, 50, 100, 50)
    assert or2 == 0.0, f"Fisher self-test FAILED: zero-cell OR={or2}, expected 0.0"
    # IT=0/50, Rest=25/150 → should be significant
    or3, p3 = fisher_exact_2x2(0, 50, 25, 125)
    assert p3 < 0.01, f"Fisher self-test FAILED: extreme table gave p={p3:.6f}, expected < 0.01"


def print_fisher_tests(sel_rates, total_n, score_dists=None):
    """Fisher's exact test: IT Services vs rest at each k."""
    print("\n" + "=" * 90)
    print("FISHER'S EXACT TEST: IT Services vs Rest-of-Groups (two-tailed)")
    print("=" * 90)
    print(f"{'k':>5}  {'IT in_k':>8} {'IT not':>8} {'Rest in_k':>10} {'Rest not':>10}  {'OR':>8} {'p-value':>10} {'sig @0.05':>10}")
    print("-" * 90)

    it = sel_rates["it_services"]
    rest_n = sum(sel_rates[gk]["n"] for gk in GROUP_KEYS if gk != "it_services")

    # Check if tie-broken artifact zone: median score 0 means ranking is index-order,
    # not score-order, making selection rates at large k meaningless.
    artifact_k = set()
    if score_dists is not None:
        zero_median_groups = [gk for gk in GROUP_KEYS if score_dists[gk]["median"] == 0.0]
        if zero_median_groups:
            artifact_k = {k for k in K_VALUES if k > 20}  # flag k > 20 when median is 0

    for k in K_VALUES:
        a = it[f"in_top_{k}"]           # IT in top-k
        b = it["n"] - a                 # IT not in top-k
        rest_in_k = sum(sel_rates[gk][f"in_top_{k}"] for gk in GROUP_KEYS if gk != "it_services")
        c = rest_in_k                   # Rest in top-k
        d = rest_n - rest_in_k          # Rest not in top-k

        odds_ratio, p_val = fisher_exact_2x2(a, b, c, d)
        if k in artifact_k:
            sig = "ARTIFACT"
        elif p_val < 0.05:
            sig = "YES *"
        else:
            sig = "no"
        print(f"{k:>5}  {a:>8} {b:>8} {c:>10} {d:>10}  {odds_ratio:>8.3f} {p_val:>10.6f} {sig:>10}")

    if artifact_k:
        print(f"\n  WARNING: k={sorted(artifact_k)} flagged as ARTIFACT: all groups have median score 0.0.")
        print("    Rankings at these k are tie-broken by index order (round-robin), not by score.")
        print("    Fisher p-values at these k are NOT valid evidence of selection bias.")


def print_sanity_honest(sel_rates, score_dists):
    """Honest sanity check — explicitly call out the asymmetry."""
    print("\n" + "=" * 90)
    print("SANITY CHECK: Disqualifier firing asymmetry (honest)")
    print("=" * 90)

    for gk in GROUP_KEYS:
        dq = score_dists[gk]["dq"]
        n = sel_rates[gk]["n"]
        print(f"  {GROUPS[gk]['label']:<20}  disq fires: {dq:>3}/{n}  ({dq/n*100:.1f}%)")

    it_dq = score_dists["it_services"]["dq"]
    it_n = sel_rates["it_services"]["n"]
    other_dq = [score_dists[gk]["dq"] for gk in GROUP_KEYS if gk != "it_services"]
    other_n = [sel_rates[gk]["n"] for gk in GROUP_KEYS if gk != "it_services"]
    total_other = sum(other_n)

    print(f"\n  IT Services:  {it_dq}/{it_n} ({it_dq/it_n*100:.1f}%)")
    print(f"  All others:   {sum(other_dq)}/{total_other} ({sum(other_dq)/total_other*100:.1f}%)")

    if total_other > 0:
        ratio = (it_dq / it_n) / (sum(other_dq) / total_other) if sum(other_dq) > 0 else float('inf')
        print(f"  Firing rate ratio (IT / others): {ratio:.1f}x")

    # Per-group detail
    print("\n  Per-group disqualifier firing detail:")
    for gk in GROUP_KEYS:
        dq = score_dists[gk]["dq"]
        n = sel_rates[gk]["n"]
        max_sc = score_dists[gk]["max"]
        mean_sc = score_dists[gk]["mean"]
        print(f"    {GROUPS[gk]['label']:<20}  fires={dq:>2}/{n:<3} ({dq/n*100:.0f}%)  max_score={max_sc:.2f}  mean_score={mean_sc:.2f}")

    # Verdict
    it_rate = it_dq / it_n if it_n else 0
    other_rate = sum(other_dq) / total_other if total_other else 0
    if it_rate > 0.8 and other_rate < 0.2:
        print(f"\n  VERDICT: DISPROPORTIONATE — IT Services disqualifier fires at {it_rate:.0%} vs {other_rate:.0%} for others.")
        print(f"  This is NOT 'broad firing across all groups'. It is a targeted {it_rate/other_rate:.1f}x asymmetry.")
    elif it_rate == 0 and other_rate == 0:
        print("\n  VERDICT: No disqualifiers fired on any group.")
    else:
        print(f"\n  VERDICT: Mixed — IT {it_rate:.0%} vs others {other_rate:.0%}.")


def print_score_ceiling(score_dists):
    """Show that IT Services max score is near other groups' means."""
    print("\n" + "=" * 90)
    print("SCORE CEILING ANALYSIS")
    print("=" * 90)

    it_max = score_dists["it_services"]["max"]
    it_mean = score_dists["it_services"]["mean"]
    other_means = [(GROUPS[gk]["label"], score_dists[gk]["mean"]) for gk in GROUP_KEYS if gk != "it_services"]
    other_medians = [(GROUPS[gk]["label"], score_dists[gk]["median"]) for gk in GROUP_KEYS if gk != "it_services"]

    print(f"  IT Services  max score:  {it_max:.2f}")
    print(f"  IT Services  mean score: {it_mean:.2f}")
    print()

    for label, mean in other_means:
        ratio_to_mean = it_max / mean if mean > 0 else float('inf')
        print(f"  {label:<20} mean: {mean:.2f}  (IT max / this mean = {ratio_to_mean:.2f}x)")

    print()
    for label, median in other_medians:
        ratio_to_median = it_max / median if median > 0 else float('inf')
        print(f"  {label:<20} median: {median:.2f}  (IT max / this median = {ratio_to_median:.2f}x)")

    # Verdict
    closest_mean = min(m for _, m in other_means) if other_means else 0
    if it_max < closest_mean * 1.2:
        print(f"\n  VERDICT: IT Services ceiling ({it_max:.2f}) is AT or BELOW other groups' means ({closest_mean:.2f}).")
        print("  This is a structural score ceiling — the disqualifier caps IT Services scores.")
    else:
        print(f"\n  VERDICT: IT Services ceiling ({it_max:.2f}) is above other groups' lowest mean ({closest_mean:.2f}).")


def vf(table, sel, sd2, rd2):
    errs = []
    gs = {gk: [r for r in table if r["group"] == gk] for gk in GROUP_KEYS}
    for gk in GROUP_KEYS:
        rs = sel_rate_multik(gs[gk]); rsc = sd(gs[gk]); rr = rd(gs[gk])
        if rs["n"] != sel[gk]["n"]:
            errs.append(f"{gk} n mismatch")
        for k in K_VALUES:
            if rs[f"in_top_{k}"] != sel[gk][f"in_top_{k}"]:
                errs.append(f"{gk} top-{k} mismatch")
        if abs(rsc["mean"] - sd2[gk]["mean"]) > 1e-6:
            errs.append(f"{gk} mean mismatch")
        if rsc["dq"] != sd2[gk]["dq"]:
            errs.append(f"{gk} disq mismatch")
        if abs(rr["mean"] - rd2[gk]["mean"]) > 1e-6:
            errs.append(f"{gk} rank_mean mismatch")
        if rr["t10"] != rd2[gk]["t10"]:
            errs.append(f"{gk} top10 mismatch")
    return errs


def run_analysis(base, parsed_jd, label="TREATMENT"):
    """Run the full analysis pipeline and return results for verification."""
    print("\n" + "=" * 100)
    print(f"{label}: round-robin employer assignment (all 4 groups in one ranking)")
    print("=" * 100)

    tcands, _ = assign_employers_round_robin(base)
    tscored = score_candidates(tcands, parsed_jd)
    ttable = build_table(tscored, {c["candidate_id"]: {"group": gk, "gl": GROUPS[gk]["label"],
                                                         "company": GROUPS[gk]["companies"][i // 50 % 4]}
                                   for i, (gk, c) in enumerate(
                                       (gk, cand)
                                       for gk in GROUP_KEYS
                                       for cand in (tcands[i] for i in range(len(tcands)))
                                   )})

    # Simpler: rebuild gmap from round-robin assignment
    gmap = {}
    for i, cand in enumerate(tcands):
        gk = GROUP_KEYS[i % 4]
        ci = (i // 4) % 4
        gmap[cand["candidate_id"]] = {"group": gk, "gl": GROUPS[gk]["label"], "company": GROUPS[gk]["companies"][ci]}
    ttable = build_table(tscored, gmap)

    for gk in GROUP_KEYS:
        rows = [r for r in ttable if r["group"] == gk]
        print(f"  {GROUPS[gk]['label']}: {len(rows)} candidates")

    tss, tsd, trd = {}, {}, {}
    for gk in GROUP_KEYS:
        rows = [r for r in ttable if r["group"] == gk]
        tss[gk] = sel_rate_multik(rows)
        tsd[gk] = sd(rows)
        trd[gk] = rd(rows)

    pt1_multi_k(tss)
    pt2(tsd)
    pt3(trd)
    pt20(ttable)

    print_fisher_tests(tss, len(ttable), score_dists=tsd)
    print_sanity_honest(tss, tsd)
    print_score_ceiling(tsd)

    errs = vf(ttable, tss, tsd, trd)
    print("\nConsistency check: " + ("FAILED: " + str(errs) if errs else "PASSED"))

    return ttable, tss, tsd, trd


def main():
    _fisher_self_test()  # validate Fisher implementation before any analysis
    print("Fisher self-test passed.")

    random.seed(SEED)
    with open(JD_PATH, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
    parsed_jd = parse_job_description(jd_input)
    print(f"Loading {N_CANDIDATES} candidates (random sample with seed={SEED})...")
    base = load_candidates(str(CANDIDATES_PATH), N_CANDIDATES)
    print(f"Loaded {len(base)} candidates.")
    report_sample_composition(base)
    base.sort(key=lambda c: c.get("candidate_id", ""))

    # ── Treatment ─────────────────────────────────────────────────────────
    ttable, tss, tsd, trd = run_analysis(base, parsed_jd, "TREATMENT")

    # ── Control ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("CONTROL: employers left untouched (same 200 candidates)")
    print("=" * 100)

    ca = {}
    for i, c in enumerate(base):
        gk = GROUP_KEYS[i % 4]; oc = "Unknown"
        ch = c.get("career_history", [])
        if ch:
            oc = ch[0].get("company", "Unknown")
        ca[c["candidate_id"]] = {"group": gk, "gl": GROUPS[gk]["label"], "company": oc}

    cscored = score_candidates(base, parsed_jd)
    ctable = build_table(cscored, ca)
    css, csd, crd = {}, {}, {}
    for gk in GROUP_KEYS:
        rows = [r for r in ctable if r["group"] == gk]
        css[gk] = sel_rate_multik(rows); csd[gk] = sd(rows); crd[gk] = rd(rows)

    pt1_multi_k(css)
    pt2(csd)
    pt3(crd)
    pt20(ctable)

    print_fisher_tests(css, len(ctable), score_dists=csd)
    print_sanity_honest(css, csd)
    print_score_ceiling(csd)

    errs2 = vf(ctable, css, csd, crd)
    print("\nControl consistency check: " + ("FAILED: " + str(errs2) if errs2 else "PASSED"))


if __name__ == "__main__":
    main()
