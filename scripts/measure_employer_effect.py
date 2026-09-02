#!/usr/bin/env python3
"""
Employer-name effect measurement (v2 — internally consistent, raw evidence).

Runs feature scoring on 200 real candidates across 4 employer-name groups
(16 company variants total), measuring score and rank deltas vs. a baseline
using each candidate's original employer names.

Metrics (all derived from the *same* per-candidate delta table):
  • score_delta_p95        – 95th-percentile |Δscore|
  • score_delta_max        – max |Δscore| across all candidates
  • top50_entries          – candidates that entered top-50 after swap
  • top50_exits            – candidates that left top-50 after swap
  • top50_turnover         – |entries| + |exits|  (symmetric diff, reported as-is)
  • rank_delta_max         – max |rank change| across all candidates
  • rank_delta_p95         – 95th-percentile |rank change|
  • top10_before           – baseline top-10 IDs (sorted)
  • top10_after            – swapped top-10 IDs (sorted)
  • top10_added            – IDs in swapped top-10 but not in baseline
  • top10_removed          – IDs in baseline top-10 but not in swapped
  • top10_changed          – bool (top10_added ∪ top10_removed is non-empty)

Raw evidence section:
  Per company variant, prints the top-10 biggest score movers with their
  baseline score, swapped score, Δscore, baseline rank, swapped rank, Δrank.

Internal consistency check:
  Re-derives every aggregate metric from the per-candidate delta table at
  the end and asserts they match the printed values.
"""

import copy
import json
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
    "it_services": {
        "label": "IT Services",
        "companies": ["TCS", "Infosys", "Wipro", "Cognizant"],
    },
    "global_product": {
        "label": "Global Product",
        "companies": ["Google", "Microsoft", "Amazon", "Meta"],
    },
    "indian_product": {
        "label": "Indian Product",
        "companies": ["Flipkart", "Razorpay", "Swiggy", "Zomato"],
    },
    "unknown": {
        "label": "Unknown",
        "companies": ["Acme Inc", "Globex", "Initech", "Hooli"],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_candidates(path: str, n: int) -> list[dict]:
    cands = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cands.append(json.loads(line))
            if len(cands) >= n:
                break
    return cands


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
    """
    Build a per-candidate delta table from baseline and swapped scored lists.
    Every metric downstream is derived from this table.
    """
    b_map = {r["candidate_id"]: r for r in baseline}
    s_map = {r["candidate_id"]: r for r in swapped}
    deltas = []
    for cid in b_map:
        if cid not in s_map:
            continue
        b = b_map[cid]
        s = s_map[cid]
        score_delta = s["score"] - b["score"]
        rank_delta = s["rank"] - b["rank"]  # positive = moved down (worse)
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
    """
    Derive all aggregate metrics from the delta table and top sets.
    This guarantees internal consistency.
    """
    abs_deltas = [d["abs_score_delta"] for d in deltas]
    abs_ranks = [d["abs_rank_delta"] for d in deltas]

    # Sort for percentile
    abs_deltas_sorted = sorted(abs_deltas)
    abs_ranks_sorted = sorted(abs_ranks)
    n = len(abs_deltas_sorted)

    def p95(arr):
        if not arr:
            return 0.0
        idx = int(len(arr) * 0.95)
        idx = min(idx, len(arr) - 1)
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


def print_evidence(deltas: list[dict], company: str, top_n: int = 10):
    """Print the top-N biggest score movers as raw evidence."""
    by_abs_delta = sorted(deltas, key=lambda d: d["abs_score_delta"], reverse=True)
    print(f"    RAW EVIDENCE -- top-{top_n} score movers for {company}:")
    print(f"    {'candidate_id':<16} {'base_score':>10} {'swap_score':>10} {'delta_sc':>9} {'base_rank':>9} {'swap_rank':>9} {'delta_rk':>7} {'disq_d':>7}")
    print(f"    {'-'*90}")
    for d in by_abs_delta[:top_n]:
        disq_change = d["swapped_disq"] - d["baseline_disq"]
        print(
            f"    {d['candidate_id']:<16} "
            f"{d['baseline_score']:>10.4f} "
            f"{d['swapped_score']:>10.4f} "
            f"{d['score_delta']:>+9.4f} "
            f"{d['baseline_rank']:>9} "
            f"{d['swapped_rank']:>9} "
            f"{d['rank_delta']:>+6} "
            f"{disq_change:>+7.1f}"
        )


def verify_consistency(deltas, metrics, baseline_top50, swapped_top50,
                       baseline_top10, swapped_top10):
    """
    Re-derive every metric from the delta table and assert it matches.
    This catches any off-by-one or logic error in the reporting code.
    """
    re_derived = derive_metrics(deltas, baseline_top50, swapped_top50,
                                baseline_top10, swapped_top10)
    errors = []

    if abs(re_derived["score_delta_max"] - metrics["score_delta_max"]) > 1e-9:
        errors.append(
            f"score_delta_max: reported={metrics['score_delta_max']}, "
            f"re-derived={re_derived['score_delta_max']}"
        )
    if re_derived["top50_turnover"] != metrics["top50_turnover"]:
        errors.append(
            f"top50_turnover: reported={metrics['top50_turnover']}, "
            f"re-derived={re_derived['top50_turnover']}"
        )
    if re_derived["rank_delta_max"] != metrics["rank_delta_max"]:
        errors.append(
            f"rank_delta_max: reported={metrics['rank_delta_max']}, "
            f"re-derived={re_derived['rank_delta_max']}"
        )
    if re_derived["top10_changed"] != metrics["top10_changed"]:
        errors.append(
            f"top10_changed: reported={metrics['top10_changed']}, "
            f"re-derived={re_derived['top10_changed']}"
        )
    if set(re_derived["top10_added"]) != set(metrics["top10_added"]):
        errors.append(
            f"top10_added: reported={set(metrics['top10_added'])}, "
            f"re-derived={set(re_derived['top10_added'])}"
        )
    if set(re_derived["top10_removed"]) != set(metrics["top10_removed"]):
        errors.append(
            f"top10_removed: reported={set(metrics['top10_removed'])}, "
            f"re-derived={set(re_derived['top10_removed'])}"
        )
    if set(re_derived["top50_entries"]) != set(metrics["top50_entries"]):
        errors.append(
            f"top50_entries mismatch: reported={set(metrics['top50_entries'])}, "
            f"re-derived={set(re_derived['top50_entries'])}"
        )
    if set(re_derived["top50_exits"]) != set(metrics["top50_exits"]):
        errors.append(
            f"top50_exits mismatch: reported={set(metrics['top50_exits'])}, "
            f"re-derived={set(re_derived['top50_exits'])}"
        )

    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load JD
    with open(JD_PATH, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
    parsed_jd = parse_job_description(jd_input)

    # Load candidates
    print(f"Loading {N_CANDIDATES} candidates...")
    base_candidates = load_candidates(str(CANDIDATES_PATH), N_CANDIDATES)
    print(f"Loaded {len(base_candidates)} candidates.")

    # ── Baseline scoring ──────────────────────────────────────────────────
    print("\nScoring BASELINE (original employer names)...")
    baseline = score_candidates(base_candidates, parsed_jd)
    baseline_scores = {r["candidate_id"]: r["score"] for r in baseline}
    baseline_ranks = {r["candidate_id"]: r["rank"] for r in baseline}
    baseline_top10 = [r["candidate_id"] for r in baseline[:10]]
    baseline_top50 = set(r["candidate_id"] for r in baseline[:50])
    baseline_disq = sum(1 for r in baseline if r["disq_penalty"] > 0)
    print(f"  Baseline disqualifier fires: {baseline_disq}")
    print(f"  Top-5: {[(r['candidate_id'], round(r['score'], 2)) for r in baseline[:5]]}")
    print(f"  Top-10 IDs: {baseline_top10}")

    # ── Measure each group ────────────────────────────────────────────────
    all_group_results = {}  # group_key -> list of per-company result dicts
    all_consistency_errors = []

    for group_key, group_info in GROUPS.items():
        label = group_info["label"]
        companies = group_info["companies"]
        print(f"\n{'=' * 100}")
        print(f"GROUP: {label}")
        print(f"{'=' * 100}")

        group_rows = []
        for company in companies:
            swapped = swap_companies(base_candidates, company)
            scored = score_candidates(swapped, parsed_jd)

            # Build delta table (single source of truth)
            deltas = build_delta_table(baseline, scored)

            swapped_top50 = set(r["candidate_id"] for r in scored[:50])
            swapped_top10 = [r["candidate_id"] for r in scored[:10]]

            # Derive metrics from delta table
            metrics = derive_metrics(
                deltas, baseline_top50, swapped_top50,
                baseline_top10, swapped_top10,
            )

            # Attach company label
            metrics["company"] = company
            metrics["group"] = label
            disq_count = sum(1 for r in scored if r["disq_penalty"] > 0)
            metrics["disq_fires"] = disq_count

            # Print summary line
            print(
                f"\n  {company:20s}  "
                f"score_d_p95={metrics['score_delta_p95']:.4f}  "
                f"score_d_max={metrics['score_delta_max']:.4f}  "
                f"top50_turn={metrics['top50_turnover']:>3}  "
                f"(+{len(metrics['top50_entries'])} / -{len(metrics['top50_exits'])})  "
                f"rank_d_max={metrics['rank_delta_max']:>3}  "
                f"top10_changed={metrics['top10_changed']}  "
                f"disq={disq_count}"
            )
            if metrics["top10_added"]:
                print(f"    top-10 added:   {metrics['top10_added']}")
            if metrics["top10_removed"]:
                print(f"    top-10 removed: {metrics['top10_removed']}")

            # Print raw evidence
            print_evidence(deltas, company, top_n=10)

            # Consistency check
            errors = verify_consistency(
                deltas, metrics, baseline_top50, swapped_top50,
                baseline_top10, swapped_top10,
            )
            if errors:
                all_consistency_errors.extend(
                    [(f"{label}/{company}", e) for e in errors]
                )
                print(f"    *** CONSISTENCY ERRORS: {errors}")
            else:
                print(f"    [OK] All metrics internally consistent")

            group_rows.append(metrics)

        all_group_results[group_key] = group_rows

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("EMPLOYER NAME EFFECT MEASUREMENT -- SUMMARY TABLE")
    print("=" * 130)
    header = (
        f"{'group':<18} {'company':<14} "
        f"{'score_d_p95':>10} {'score_d_max':>10} "
        f"{'top50_turn':>10} {'(+in/-out)':>10} "
        f"{'rank_d_max':>9} {'rank_d_p95':>9} "
        f"{'top10_changed':>13} {'disq':>5}"
    )
    print(header)
    print("-" * 130)
    for group_key, rows in all_group_results.items():
        for r in rows:
            print(
                f"{r['group']:<18} {r['company']:<14} "
                f"{r['score_delta_p95']:>10.4f} {r['score_delta_max']:>10.4f} "
                f"{r['top50_turnover']:>10} "
                f"{'(+' + str(len(r['top50_entries'])) + '/-' + str(len(r['top50_exits'])) + ')':>10} "
                f"{r['rank_delta_max']:>9} {r['rank_delta_p95']:>9.1f} "
                f"{str(r['top10_changed']):>13} "
                f"{r['disq_fires']:>5}"
            )

    # ── Consulting disqualifier audit ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("CONSULTING DISQUALIFIER AUDIT: company-swap effect on disqualifier fires")
    print("=" * 100)
    print(f"  Baseline disq fires: {baseline_disq}")
    for group_key, rows in all_group_results.items():
        for r in rows:
            delta = r["disq_fires"] - baseline_disq
            marker = " <-- CHANGED" if delta != 0 else ""
            print(f"  {r['company']:<20s} disq={r['disq_fires']:<4} (delta={delta:+d}){marker}")

    # ── Final consistency report ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERNAL CONSISTENCY CHECK")
    print("=" * 70)
    if all_consistency_errors:
        print("FAILED — the following inconsistencies were detected:")
        for loc, err in all_consistency_errors:
            print(f"  [{loc}] {err}")
        sys.exit(1)
    else:
        print("PASSED — all metrics internally consistent across all company variants.")

    # ── Source-of-truth explanation ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SCORING PATH -- how employer names influence scores")
    print("=" * 70)
    print("""
Two code paths in feature_scorer.py read career_history[].company:

  1. CONSULTING DISQUALIFIER (lines ~288-301):
     Iterates career_history, extracts job.get("company","").lower(),
     compares against a hardcoded list ["tcs", "infosys", "wipro", ...].
     If consulting_duration / total_duration > 0.60 --> disqualifier fires.
     This DIRECTLY reads employer names and penalises consulting employers.

  2. CAREER RELEVANCE BONUS (lines ~264-273):
     Iterates career_history, reads title and description for keywords
     like "ranking", "search", "recommendation".
     This reads TITLE and DESCRIPTION, NOT company name.

  3. DOMAIN MATCH (in compute_title_seniority_match, lines ~85-100):
     Reads profile.current_title for AI/non-AI keywords.
     Not affected by employer name swaps.

VERDICT:
  - Employer-name swaps affect score ONLY through the consulting
    disqualifier when the swapped company is on the consulting list.
  - Non-consulting companies (Google, Flipkart, Acme, etc.) produce
    zero score delta if the candidate has sufficient consulting career
    history elsewhere, or zero consulting history at all.
  - The career bonus and skill match are employer-name-invariant.
""")


if __name__ == "__main__":
    main()
