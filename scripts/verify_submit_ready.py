"""Pre-submission checklist — run before portal upload."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "mohd_ibadullah.csv"
XLSX = ROOT / "outputs" / "mohd_ibadullah.xlsx"
META = ROOT / "submission_metadata.yaml"
EMB = ROOT / "data" / "candidate_embeddings.npy"
IDS = ROOT / "data" / "candidate_ids.json"
VALIDATOR = ROOT.parent / "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.honeypot_detector import detect_trap


def check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return ok


def main() -> int:
    print("TalentLens AI — Submit Readiness Check\n")
    all_ok = True

    all_ok &= check("CSV exists", CSV.exists(), str(CSV))
    all_ok &= check("XLSX exists", XLSX.exists(), str(XLSX))

    if CSV.exists():
        vpath = VALIDATOR if VALIDATOR.exists() else None
        if vpath:
            r = subprocess.run([sys.executable, str(vpath), str(CSV)], capture_output=True, text=True)
            all_ok &= check("Validator", r.returncode == 0 and "valid" in r.stdout.lower(), r.stdout.strip() or r.stderr.strip())
        rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
        all_ok &= check("Row count", len(rows) == 100, f"{len(rows)} rows")

    all_ok &= check("Embeddings cached", EMB.exists() and IDS.exists(), "data/candidate_embeddings.npy + candidate_ids.json")

    if META.exists():
        text = META.read_text(encoding="utf-8")
        bad_phone = "+91 99999 99999" in text or 'phone: ""' in text
        all_ok &= check("Phone not placeholder", not bad_phone, "Update submission_metadata.yaml + portal with real number")

    if CSV.exists() and (ROOT.parent / "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl").exists():
        cand_path = ROOT.parent / "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
        ids_top = {r["candidate_id"] for r in rows}
        cmap = {}
        with open(cand_path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if c["candidate_id"] in ids_top:
                    cmap[c["candidate_id"]] = c
        traps = sum(1 for cid in ids_top if detect_trap(cmap[cid])[0] > 0.4)
        all_ok &= check("Honeypots in top 100", traps == 0, f"{traps} found")

    print()
    if all_ok:
        print("READY — upload PDF, XLSX, GitHub link, and real phone to portal.")
        return 0
    print("NOT READY — fix FAIL items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
