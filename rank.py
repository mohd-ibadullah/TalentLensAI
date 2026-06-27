"""
TalentLens AI — official ranking entry point (alias for run_pipeline_full.py).
Usage: python rank.py --candidates ./candidates.jsonl --out ./outputs/mohd_ibadullah.csv
"""
from src.run_pipeline_full import main

if __name__ == "__main__":
    main()
