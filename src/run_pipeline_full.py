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
from src.preflight import run_preflight, ensure_embeddings_exist


def run_setup(project_root: Path) -> None:
    """One-time: download models + precompute embeddings."""
    dl = project_root / "src" / "download_models.py"
    pc = project_root / "src" / "precompute_embeddings.py"
    for script in (dl, pc):
        print(f"\n>>> Running {script.name}...")
        subprocess.run([sys.executable, str(script)], check=True)


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
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip model-cache preflight (not recommended for first run)")
    parser.add_argument("--check-pool", action="store_true",
                        help="Verify all candidate_ids exist in candidates.jsonl")
    parser.add_argument("--setup", action="store_true",
                        help="Run one-time model download + embedding precompute, then exit")
    parser.add_argument("--allow-bm25-only", action="store_true",
                        help="Allow ranking without precomputed embeddings (lower quality)")
    args = parser.parse_args()

    if args.setup:
        run_setup(project_root)
        print("\nSetup finished. Re-run without --setup to rank.")
        return

    candidates_path = args.candidates
    jd_config_path = args.jd
    output_csv_path = args.out

    if not os.path.exists(jd_config_path):
        print(f"Error: JD config file not found at {jd_config_path}")
        sys.exit(1)

    is_jsonl = candidates_path.lower().endswith(".jsonl")
    if is_jsonl and not args.allow_bm25_only and not ensure_embeddings_exist(verbose=True):
        print("\nERROR: Precomputed embeddings missing. Full hybrid ranking requires them.")
        print("  Fix: python src/run_pipeline_full.py --setup")
        print("  Or:  .\\setup.ps1")
        sys.exit(1)

    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)

    print(f"Loaded Job Description: {jd_input['role_title']}")
    print(f"Running pipeline on dataset: {candidates_path}")

    if not args.skip_preflight:
        run_preflight(require_embeddings=False, verbose=True)
    
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

    if args.check_pool and os.path.exists(candidates_path):
        import csv
        pool_ids = set()
        with open(candidates_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pool_ids.add(json.loads(line)["candidate_id"])
        with open(output_csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        bad = [r["candidate_id"] for r in rows if r["candidate_id"] not in pool_ids]
        if bad:
            print(f"ERROR: {len(bad)} candidate_ids not in pool: {bad[:5]}")
            sys.exit(1)
        print(f"Pool check passed: all {len(rows)} IDs exist in candidates.jsonl")

if __name__ == "__main__":
    main()
