"""
Honeypot detector evaluation on sample candidates.
Usage: python src/test_honeypot.py --data ./data/sample_candidates.json
"""
import json
import os
import sys
import argparse
from pathlib import Path

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.honeypot_detector import detect_trap


def analyze_honeypots():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Evaluate honeypot detector")
    parser.add_argument("--data", default=str(project_root / "data" / "sample_candidates.json"),
                        help="Path to sample_candidates.json")
    args = parser.parse_args()

    data_path = args.data
    with open(data_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Let's inspect specific candidates
    target_ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004", "CAND_0000005"]
    
    print("--- Honeypot Detector Evaluation ---")
    for c in candidates:
        if c["candidate_id"] in target_ids:
            score, reason = detect_trap(c)
            print(f"\nCandidate: {c['candidate_id']} | Name: {c['profile']['anonymized_name']}")
            print(f"Current Title: {c['profile']['current_title']}")
            print(f"Trap Score: {score:.4f}")
            print(f"Trap Reason: {reason}")

if __name__ == "__main__":
    analyze_honeypots()
