"""
Test pipeline on sample candidates.
Usage: python src/test_pipeline.py --data ./data/sample_candidates.json
"""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import run_ranking_pipeline


def test():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Test pipeline on sample data")
    parser.add_argument("--data", default=str(project_root / "data" / "sample_candidates.json"),
                        help="Path to sample_candidates.json")
    parser.add_argument("--jd", default=str(project_root / "config" / "job_description.json"),
                        help="Path to job_description.json")
    parser.add_argument("--out", default=str(project_root / "outputs" / "sample_test_run.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    candidates_path = args.data
    jd_config_path = args.jd
    out_csv_path = args.out

    # Load JD
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
        
    print("Running test pipeline on sample candidates...")
    top_candidates = run_ranking_pipeline(
        candidates_path=candidates_path,
        jd_input=jd_input,
        out_csv_path=out_csv_path,
        top_n=10,
        use_llm=False
    )
    
    print("\n--- Pipeline Ranking Output (Top 10) ---")
    for cand in top_candidates:
        print(f"Rank {cand['_rank']}: {cand['candidate_id']} | Score: {cand['_final_score']:.2f}")
        print(f"  Title: {cand['profile']['current_title']} | Trap Score: {cand['_trap_score']:.2f}")
        print(f"  Breakdown: Sim={cand['_breakdown']['semantic_similarity']:.2f}, Skill={cand['_breakdown']['skill_match_score']:.2f}, Title={cand['_breakdown']['title_seniority_match']:.2f}, Signals={cand['_breakdown']['signal_bonus']:.2f}")
        print(f"  Reasoning: {cand['_reasoning']}")
        print("-" * 50)
        
    # Check if CAND_0000001 is ranked higher than CAND_0000002
    ranks = {cand["candidate_id"]: cand["_rank"] for cand in top_candidates}
    if "CAND_0000001" in ranks:
        print(f"\nSUCCESS: CAND_0000001 (genuine AI) is ranked at #{ranks['CAND_0000001']}")
    else:
        print("\nNote: CAND_0000001 is not in top 10.")
        
    for decoy_id in ["CAND_0000002", "CAND_0000003", "CAND_0000004", "CAND_0000005"]:
        if decoy_id in ranks:
            print(f"WARNING: Decoy {decoy_id} is in the top 10! Rank: #{ranks[decoy_id]}")
        else:
            print(f"SUCCESS: Decoy {decoy_id} was successfully filtered out of the top 10.")

if __name__ == "__main__":
    test()
