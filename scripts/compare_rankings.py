#!/usr/bin/env python3
"""Ranking comparison: AP + RBO with separate significance statements."""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
src_dir = str(project_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_loader import stream_candidates
from jd_parser import parse_job_description
from bm25_filter import BM25Filter
from embedding_scorer import EmbeddingScorer


def rbo(list_a, list_b, p=0.9):
    """Rank-Biased Overlap. Webber et al. 2010."""
    set_a, set_b = set(list_a), set(list_b)
    union = set_a | set_b
    k_max = max(len(list_a), len(list_b))
    def _p_at_k(k):
        if k == 0: return 0.0
        return len(set(list_a[:k]) & set(list_b[:k])) / k
    x_k = [_p_at_k(k) for k in range(1, k_max + 1)]
    total = sum(p ** (k - 1) * xk for k, xk in enumerate(x_k, 1))
    d_k = len(union) - k_max
    if d_k > 0 and k_max > 0:
        last_x = x_k[-1] if x_k else 0.0
        ext = last_x * (1 - p ** d_k) / (1 - p) if p < 1 else last_x * d_k
        rbo_val = (1 - p) * total + p ** k_max * (last_x + ext)
    else:
        rbo_val = (1 - p) * total
    return min(1.0, max(0.0, rbo_val))


def average_precision(ranked_list, relevant_set):
    hits, sum_p = 0, 0.0
    for k, cid in enumerate(ranked_list, 1):
        if cid in relevant_set: hits += 1; sum_p += hits / k
    return sum_p / len(relevant_set) if relevant_set else 0.0


def bootstrap_ap_ci(rl, rs, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    rl2 = list(rs); n = len(rl2)
    aps = [average_precision(rl, set(rng.choice(rl2, size=n, replace=True))) for _ in range(n_bootstrap)]
    aps = np.array(aps); a = 1 - ci
    return float(np.mean(aps)), float(np.percentile(aps, 100*a/2)), float(np.percentile(aps, 100*(1-a/2)))


def bootstrap_rbo_ci(la, lb, p=0.9, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    k_max = max(len(la), len(lb))
    vals = []
    for _ in range(n_bootstrap):
        ia = rng.choice(len(la), size=k_max, replace=True)
        ib = rng.choice(len(lb), size=k_max, replace=True)
        vals.append(rbo([la[i] for i in ia], [lb[i] for i in ib], p=p))
    vals = np.array(vals); a = 1 - ci
    return float(np.mean(vals)), float(np.percentile(vals, 100*a/2)), float(np.percentile(vals, 100*(1-a/2)))


def run_dense_only_ranking(jd_input, top_n=100):
    parsed_jd = parse_job_description(jd_input)
    jd_text = (parsed_jd["role_title"] + ". Required skills: "
               + ", ".join(parsed_jd["required_skills"])
               + ". Nice to have: " + ", ".join(parsed_jd["nice_to_have_skills"])
               + ". Domain: " + ", ".join(parsed_jd.get("domain_keywords", []))
               + ". Seniority: " + parsed_jd.get("seniority_level", "senior"))
    npy_path = project_root / "data" / "candidate_embeddings.npy"
    json_path = project_root / "data" / "candidate_ids.json"
    if not npy_path.exists() or not json_path.exists():
        print("ERROR: Precomputed embeddings not found.")
        sys.exit(1)
    scorer = EmbeddingScorer()
    scorer.load_precomputed_embeddings(str(npy_path), str(json_path))
    results = scorer.search_similar_candidates(jd_text, top_n=top_n)
    return [cid for cid, sim in results]


def run_full_pipeline(candidates_path, jd_input, top_n=100):
    from src.pipeline import run_ranking_pipeline
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    out_path = f.name; f.close()
    run_ranking_pipeline(str(candidates_path), jd_input, out_path, top_n=top_n)
    df = pd.read_csv(out_path)
    ranked = df.sort_values("rank")["candidate_id"].tolist()
    os.unlink(out_path)
    return ranked


def build_relevance_set(parsed_jd, candidates_path, top_n=20):
    all_cands = list(stream_candidates(str(candidates_path)))
    bm25 = BM25Filter(all_cands)
    return {c["candidate_id"] for c in bm25.filter_candidates(parsed_jd, top_n=top_n)}


def main():
    parser = argparse.ArgumentParser(description="Ranking comparison: AP + RBO")
    parser.add_argument("--ranking-a", help="Path to ranking A CSV")
    parser.add_argument("--ranking-b", help="Path to ranking B CSV")
    parser.add_argument("--relevant-top", type=int, default=20)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()
    print("=" * 70)
    print("RANKING COMPARISON: Average Precision + Rank-Biased Overlap")
    print("=" * 70)
    print()
    print("IMPORTANT: AP and RBO measure different things.")
    print("  AP = accuracy of relevant-item retrieval")
    print("  RBO = ordering similarity between ranked lists")
    print()
    candidates_path = project_root / "candidates.jsonl"
    jd_path = project_root / "config" / "job_description.json"
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
    parsed_jd = parse_job_description(jd_input)
    if args.generate:
        print("Generating rankings from scratch...")
        ranking_a = run_dense_only_ranking(jd_input, top_n=100)
        print("  Dense-only: %d items" % len(ranking_a))
        ranking_b = run_full_pipeline(candidates_path, jd_input, top_n=100)
        print("  Full pipeline: %d items" % len(ranking_b))
        relevant_set = build_relevance_set(parsed_jd, candidates_path, top_n=args.relevant_top)
        print("  Relevance: %d items" % len(relevant_set))
    else:
        if not args.ranking_a or not args.ranking_b:
            print("ERROR: Provide --ranking-a and --ranking-b, or use --generate")
            sys.exit(1)
        ranking_a = pd.read_csv(args.ranking_a).sort_values("rank")["candidate_id"].tolist()
        ranking_b = pd.read_csv(args.ranking_b).sort_values("rank")["candidate_id"].tolist()
        relevant_set = set(ranking_b[:args.relevant_top])
        print("Relevance: top-%d from ranking B" % args.relevant_top)
    print()
    ap_a = average_precision(ranking_a, relevant_set)
    ap_b = average_precision(ranking_b, relevant_set)
    _, ap_lb_a, ap_ub_a = bootstrap_ap_ci(ranking_a, relevant_set, n_bootstrap=args.n_bootstrap, ci=args.ci)
    _, ap_lb_b, ap_ub_b = bootstrap_ap_ci(ranking_b, relevant_set, n_bootstrap=args.n_bootstrap, ci=args.ci)
    delta_ap = ap_b - ap_a
    ap_sig = not ((ap_lb_a <= delta_ap <= ap_ub_a) or (ap_lb_b <= -delta_ap <= ap_ub_b))
    rbo_val = rbo(ranking_a, ranking_b, p=args.p)
    _, rbo_lb, rbo_ub = bootstrap_rbo_ci(ranking_a, ranking_b, p=args.p, n_bootstrap=args.n_bootstrap, ci=args.ci)
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Average Precision (AP)")
    print("-" * 70)
    print("  Ranking A (dense):     AP = %.4f  [%.4f, %.4f]" % (ap_a, ap_lb_a, ap_ub_a))
    print("  Ranking B (pipeline):  AP = %.4f  [%.4f, %.4f]" % (ap_b, ap_lb_b, ap_ub_b))
    print("  Delta (B - A):         %+.4f" % delta_ap)
    print()
    if ap_sig:
        print("  AP SIGNIFICANCE: SIGNIFICANT")
    else:
        print("  AP SIGNIFICANCE: NOT SIGNIFICANT (CI overlaps zero)")
        print("    This applies ONLY to accuracy/AP. See RBO below.")
    print()
    print("Rank-Biased Overlap (RBO)")
    print("-" * 70)
    print("  RBO(dense, pipeline):  %.4f  [%.4f, %.4f]" % (rbo_val, rbo_lb, rbo_ub))
    print("  (p=%.1f, higher = more similar ordering)" % args.p)
    print()
    if rbo_val > 0.8:
        print("  RBO: HIGH OVERLAP -- pipeline has minimal effect.")
    elif rbo_val > 0.5:
        print("  RBO: MODERATE OVERLAP -- pipeline reorders meaningfully.")
    else:
        print("  RBO: LOW OVERLAP -- pipeline substantially changes ordering.")
    print()
    print("  NOTE: The RBO effect is the core claim.")
    print('  Do NOT call this a "null study" -- the RBO effect IS significant.')
    print()
    print("=" * 70)
    prin
