"""
TalentLens AI — official ranking entry point (alias for run_pipeline_full.py).
Usage: python rank.py --candidates ./candidates.jsonl --out ./outputs/sample_ranking_output.csv
"""
from src.run_pipeline_full import main

if __name__ == "__main__":
    main()
