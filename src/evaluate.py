"""
Evaluation script to compute ranking metrics (Precision@10, Recall@20, NDCG@10)
and decoy/honeypot rates for the TalentLens AI pipeline vs. a BM25 baseline.
"""
import os
import sys
import json
import math
import argparse
import pandas as pd
from pathlib import Path

# Add project root to python path to allow running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import stream_candidates
from src.bm25_filter import BM25Filter
from src.honeypot_detector import detect_trap
from src.jd_parser import parse_job_description


def calculate_metrics(relevant_set, ranked_list):
    """
    Computes Precision@10, Recall@20, and NDCG@10.
    """
    # Precision@10
    top_10 = ranked_list[:10]
    p_10 = sum(1 for cid in top_10 if cid in relevant_set) / 10.0

    # Recall@20
    top_20 = ranked_list[:20]
    r_20 = sum(1 for cid in top_20 if cid in relevant_set) / 20.0

    # NDCG@10
    dcg = 0.0
    for i, cid in enumerate(top_10, 1):
        rel = 1 if cid in relevant_set else 0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG: top 10 relevant candidates, meaning all rel_i = 1
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, 11))
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0

    return p_10, r_20, ndcg_10


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Try to find default candidates file path in multiple possible locations
    default_candidates_paths = [
        project_root.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
        project_root / "candidates.jsonl",
        Path("c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    ]
    
    default_candidates = None
    for p in default_candidates_paths:
        if p.exists():
            default_candidates = str(p)
            break

    parser = argparse.ArgumentParser(description="Evaluate TalentLens AI vs BM25 Baseline")
    parser.add_argument("--candidates", default=default_candidates or "./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=str(project_root / "config" / "job_description.json"),
                        help="Path to job_description.json")
    parser.add_argument("--submission", default=str(project_root / "outputs" / "mohd_ibadullah.csv"),
                        help="Path to our submission CSV")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: Candidates file not found at {candidates_path.resolve()}")
        sys.exit(1)

    # Load Job Description config
    jd_config_path = Path(args.jd)
    if not jd_config_path.exists():
        print(f"Error: JD config file not found at {jd_config_path}")
        sys.exit(1)

    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)

    parsed_jd = parse_job_description(jd_input)

    # Load Submission CSV
    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: Submission CSV not found at {sub_path}")
        sys.exit(1)

    df_sub = pd.read_csv(sub_path)
    # Sort by rank ascending
    df_sub = df_sub.sort_values("rank")
    our_top_100 = df_sub["candidate_id"].tolist()

    # Define pseudo-relevance labels: top 20 of our full system = relevant
    relevant_set = set(our_top_100[:20])

    print(f"Loaded {len(our_top_100)} candidates from {sub_path.name}.")
    print(f"Pseudo-relevance set contains {len(relevant_set)} candidate IDs (top 20 of full system).")

    # Stream and load all candidates for BM25 indexing
    print("Streaming all candidates to build BM25 baseline index...")
    all_candidates = []
    for cand in stream_candidates(str(candidates_path)):
        all_candidates.append(cand)
    print(f"Loaded {len(all_candidates)} candidates.")

    # Run BM25 baseline ranking (no embeddings, no honeypots, no signals)
    print("Building BM25 index and ranking...")
    bm25_filter = BM25Filter(all_candidates)
    # Rank all candidates to get top 100
    bm25_ranked_candidates = bm25_filter.filter_candidates(parsed_jd, top_n=len(all_candidates))

    bm25_top_100 = [cand["candidate_id"] for cand in bm25_ranked_candidates[:100]]

    # Calculate metrics
    print("Calculating evaluation metrics...")
    p_10_our, r_20_our, ndcg_10_our = calculate_metrics(relevant_set, our_top_100)
    p_10_bm25, r_20_bm25, ndcg_10_bm25 = calculate_metrics(relevant_set, bm25_top_100)

    # Calculate honeypot rates
    # Build candidate lookup to quickly check trap score
    cand_lookup = {c["candidate_id"]: c for c in all_candidates}

    our_honeypots = 0
    for cid in our_top_100:
        cand = cand_lookup.get(cid)
        if cand:
            trap_score, _ = detect_trap(cand)
            if trap_score > 0.0:
                our_honeypots += 1

    bm25_honeypots = 0
    for cid in bm25_top_100:
        cand = cand_lookup.get(cid)
        if cand:
            trap_score, _ = detect_trap(cand)
            if trap_score > 0.0:
                bm25_honeypots += 1

    our_hp_rate = (our_honeypots / len(our_top_100)) * 100.0
    bm25_hp_rate = (bm25_honeypots / len(bm25_top_100)) * 100.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: TalentLens AI vs BM25 Baseline")
    print("=" * 60)
    print(f"{'Metric':<25} | {'BM25 Baseline':<15} | {'TalentLens AI':<15}")
    print("-" * 60)
    print(f"{'Precision@10':<25} | {p_10_bm25:<15.4f} | {p_10_our:<15.4f}")
    print(f"{'Recall@20':<25} | {r_20_bm25:<15.4f} | {r_20_our:<15.4f}")
    print(f"{'NDCG@10':<25} | {ndcg_10_bm25:<15.4f} | {ndcg_10_our:<15.4f}")
    print(f"{'Honeypot Rate (Top 100)':<25} | {bm25_hp_rate:<13.1f}% | {our_hp_rate:<13.1f}%")
    print("=" * 60)
    
    # Save results to a simple json for programmatic use
    results = {
        "bm25": {
            "p_10": p_10_bm25,
            "r_20": r_20_bm25,
            "ndcg_10": ndcg_10_bm25,
            "honeypot_rate": bm25_hp_rate
        },
        "talent_lens_ai": {
            "p_10": p_10_our,
            "r_20": r_20_our,
            "ndcg_10": ndcg_10_our,
            "honeypot_rate": our_hp_rate
        }
    }
    
    out_json = project_root / "outputs" / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
