import json
import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import run_ranking_pipeline

def main():
    # Define absolute paths
    candidates_path = r"c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    jd_config_path = r"c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/talent-lens-ai/config/job_description.json"
    
    # Output file name: registered participant ID
    output_filename = "mohd_ibadullah.csv"
    output_dir = r"c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/talent-lens-ai/outputs"
    output_csv_path = os.path.join(output_dir, output_filename)
    
    # Challenge verification directory
    challenge_dir = r"c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge"
    challenge_csv_path = os.path.join(challenge_dir, output_filename)
    validator_path = os.path.join(challenge_dir, "validate_submission.py")
    
    # Load Job Description config
    if not os.path.exists(jd_config_path):
        print(f"Error: JD config file not found at {jd_config_path}")
        return
        
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
        
    print(f"Loaded Job Description: {jd_input['role_title']}")
    print(f"Running pipeline on full dataset (100K profiles)...")
    
    # Run the ranking pipeline
    # We will get the top 100 candidates
    top_candidates = run_ranking_pipeline(
        candidates_path=candidates_path,
        jd_input=jd_input,
        out_csv_path=output_csv_path,
        top_n=100,
        use_llm=False
    )
    
    # Also write a copy to the challenge directory for validate_submission.py to access easily
    print(f"Copying output to challenge folder: {challenge_csv_path}")
    import shutil
    shutil.copy(output_csv_path, challenge_csv_path)
    
    # Run validate_submission.py
    print(f"\nRunning validation script: python {validator_path} {challenge_csv_path} ...")
    try:
        result = subprocess.run(
            ["python", validator_path, challenge_csv_path],
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

if __name__ == "__main__":
    main()
