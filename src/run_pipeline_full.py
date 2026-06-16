"""
Full pipeline runner for production ranking.
Usage: python src/run_pipeline_full.py --candidates ./candidates.jsonl --jd ./config/job_description.json --out ./outputs/mohd_ibadullah.csv
"""
import json
import os
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import run_ranking_pipeline


def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Run full TalentLens AI pipeline")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=str(project_root / "config" / "job_description.json"),
                        help="Path to job_description.json")
    parser.add_argument("--out", default=str(project_root / "outputs" / "mohd_ibadullah.csv"),
                        help="Output CSV path")
    parser.add_argument("--validate", default=None,
                        help="Path to validate_submission.py (optional)")
    args = parser.parse_args()

    candidates_path = args.candidates
    jd_config_path = args.jd
    output_csv_path = args.out

    # Load Job Description config
    if not os.path.exists(jd_config_path):
        print(f"Error: JD config file not found at {jd_config_path}")
        return
        
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
        
    print(f"Loaded Job Description: {jd_input['role_title']}")
    print(f"Running pipeline on dataset: {candidates_path}")
    
    # Run the ranking pipeline
    top_candidates = run_ranking_pipeline(
        candidates_path=candidates_path,
        jd_input=jd_input,
        out_csv_path=output_csv_path,
        top_n=100,
        use_llm=False
    )
    
    # Run validation if validator path provided
    if args.validate and os.path.exists(args.validate):
        print(f"\nRunning validation script: python {args.validate} {output_csv_path} ...")
        try:
            result = subprocess.run(
                ["python", args.validate, output_csv_path],
                capture_output=True,
                text=True,
                check=True
            )
            print("Validation Result:")
            print(result.stdout)
            print("SUCCESS: Submission CSV is fully valid!")
        except subprocess.CalledProcessError as e:
            print("ERROR: Validation failed!")
            print(e.stderr)
            print(e.stdout)
            sys.exit(1)
    else:
        print(f"\nOutput saved to: {output_csv_path}")
        print("Run validate_submission.py manually to verify.")

if __name__ == "__main__":
    main()
