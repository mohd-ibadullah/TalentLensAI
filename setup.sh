#!/usr/bin/env bash
# One-time judge setup (network required). Ranking after this is offline (~30s).
set -euo pipefail
cd "$(dirname "$0")"
echo "=== TalentLens AI Setup ==="
pip install -r requirements.txt
python src/download_models.py
python src/precompute_embeddings.py
echo ""
echo "Setup complete. Run ranking:"
echo "  python rank.py --candidates <path-to-candidates.jsonl> --out ./outputs/mohd_ibadullah.csv"
