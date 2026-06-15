import argparse
import json
import os
import sys

# Add directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import run_ranking_pipeline

def main():
    parser = argparse.ArgumentParser(description="TalentLens AI Candidate Ranker CLI")
    parser.add_argument(
        "--candidates", 
        type=str, 
        default="./candidates.jsonl",
        help="Path to candidates.jsonl file"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="./mohd_ibadullah.csv",
        help="Path to output ranked CSV"
    )
    
    args = parser.parse_args()
    
    # Locate Job Description config
    jd_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "job_description.json"
    )
    
    if not os.path.exists(jd_config_path):
        print(f"Error: JD config file not found at {jd_config_path}")
        sys.exit(1)
        
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
        
    print(f"Running TalentLens AI on Candidates: {args.candidates}")
    print(f"Saving Output Ranked List to: {args.out}")
    
    # Run the pipeline
    run_ranking_pipeline(
        candidates_path=args.candidates,
        jd_input=jd_input,
        out_csv_path=args.out,
        top_n=100,
        use_llm=False
    )
    
    print("Execution complete.")

if __name__ == "__main__":
    main()
