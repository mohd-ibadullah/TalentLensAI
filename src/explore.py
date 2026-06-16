"""
Data exploration script for candidate profiles.
Usage: python src/explore.py --data ./path/to/sample_candidates.json
"""
import json
import os
import argparse
import pandas as pd
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Explore candidate dataset")
    parser.add_argument("--data", default="./data/sample_candidates.json",
                        help="Path to sample_candidates.json")
    args = parser.parse_args()

    data_path = args.data

    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    print(f"Loaded {len(candidates)} candidates from sample data.")
    print("Schema representation of first candidate:")
    c1 = candidates[0]
    print(json.dumps(c1, indent=2)[:1000])
    
    print("\n--- Basic Profile Distributions ---")
    titles = [c["profile"]["current_title"] for c in candidates]
    countries = [c["profile"]["country"] for c in candidates]
    exp_years = [c["profile"]["years_of_experience"] for c in candidates]
    
    print(f"Unique current titles: {len(set(titles))}")
    print("Top 10 titles:")
    for title, count in Counter(titles).most_common(10):
        print(f"  - {title}: {count}")
        
    print("\nExperience distribution:")
    df_exp = pd.Series(exp_years)
    print(df_exp.describe())
    
    print("\n--- Redrob Signals distributions ---")
    git_scores = [c["redrob_signals"].get("github_activity_score", -1) for c in candidates]
    recruiter_resp = [c["redrob_signals"].get("recruiter_response_rate", -1) for c in candidates]
    
    print(f"GitHub activity scores (-1 = missing):")
    print(pd.Series(git_scores).value_counts().head())
    
    print(f"\nRecruiter response rates:")
    print(pd.Series(recruiter_resp).describe())
    
    print("\n--- Decoy Analysis ---")
    marketing_summary_count = 0
    decoy_examples = []
    
    for c in candidates:
        summary = c["profile"]["summary"]
        title = c["profile"]["current_title"]
        cid = c["candidate_id"]
        # Look for standard decoy summary containing marketing manager or HR manager references
        if "marketing manager" in summary.lower() or "hr manager" in summary.lower():
            marketing_summary_count += 1
            if len(decoy_examples) < 3:
                decoy_examples.append((cid, title, summary[:120]))
                
    print(f"Candidates with 'marketing manager' or 'hr manager' summary text: {marketing_summary_count} ({marketing_summary_count/len(candidates)*100:.1f}%)")
    print("Examples of such candidates:")
    for cid, title, summ in decoy_examples:
        print(f"  - {cid} ({title}): {summ}...")

if __name__ == "__main__":
    main()
