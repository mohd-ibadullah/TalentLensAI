# Precomputed embeddings (required for full 100K ranking)

These files are **not committed to git** (~300 MB). Generate once before running the pipeline:

```bash
pip install -r requirements.txt
python src/download_models.py
python src/precompute_embeddings.py
```

Or on Windows: `.\setup.ps1`  
Or on Linux/Mac: `./setup.sh`

Place `candidates.jsonl` in the challenge bundle path (see `precompute_embeddings.py` search paths).

After generation you should have:
- `data/candidate_embeddings.npy` — 100000 × 768 float32 matrix
- `data/candidate_ids.json` — ID ordering matching the jsonl file
