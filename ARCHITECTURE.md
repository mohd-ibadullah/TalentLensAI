# TalentLens AI — System Architecture

## Pipeline Overview
```
Job Description (text/JSON)
         ↓
[Stage 1] Hybrid Retrieval (BM25 + Dense)
100,000 candidates → ~1,500 union recall (two-pass streaming, mmap embeddings)
         ↓
[Stage 2] Honeypot Trap Detector
Flags career-title mismatches, keyword stuffing, template profiles, timeline traps
         ↓
[Stage 3] Semantic + Feature Scoring
BAAI/bge-base-en-v1.5 similarity + weighted features
Semantic(40%) + Skills(20%) + Title/YoE(20%) + Signals(10%), trap veto at ≥0.40
         ↓
[Stage 4] Cross-Encoder Reranker
cross-encoder/ms-marco-MiniLM-L6-v2 on top 150 — 60% feature + 40% CE blend
         ↓
[Stage 5] Profile-Specific Reasoning (offline, rule-based)
Rank-aware summaries citing skills, company, Redrob signals — no API calls
         ↓
[Stage 6] CSV Output
Top 100 ranked candidates — validator PASS, monotonic scores
         ↓
Output: mohd_ibadullah.csv — ~28 seconds warm (models + embeddings cached)
```

## Key Design Decisions
- Local models only during ranking (`use_llm=False`) — CPU-safe, offline-reproducible
- Two-pass JSONL streaming — BM25 index + lazy load ~1,589 recall profiles (not all 100K in RAM)
- Precomputed embeddings (one-time setup) — hybrid dense recall without runtime encoding of 100K
- Honeypot detection with continuous trap scores — 0% honeypots in top 100
- Cross-encoder reranker on top 150 — precision boost without blowing the 5-minute budget
- `-1` sentinel values treated as neutral (not penalized) across all signal fields

## Performance Benchmarks
| Metric | Value |
|--------|-------|
| Total Runtime (100K, warm) | **~28 seconds** |
| First run (model download + no cache) | up to ~8 min — run `setup.ps1` once before judging |
| Peak Memory | **~2–4 GB** (under 16 GB budget) |
| Stage 5 Cross-Encoder Rerank | ~3 seconds |
| Honeypot Rate in Top 100 | 0.0% |
| CSV Validation | PASS |

## System Evaluation (TalentLens AI vs BM25 Baseline)
Pseudo-relevance labels from top-20 pipeline output — measures **relative lift**, not ground-truth accuracy:

* **Precision@10:** +150.0% relative lift vs BM25 baseline
* **Recall@20:** +150.0% relative lift vs BM25 baseline
* **NDCG@10:** +133.2% relative lift vs BM25 baseline
* **Adversarial:** BM25 top-1000 has ~30.1% honeypots; our Stage 2 filters all decoys from final top 100

## One-Time Setup (before Stage 3 reproduction)
```bash
pip install -r requirements.txt
python src/download_models.py
python src/precompute_embeddings.py
# Or: .\setup.ps1  (Windows) / ./setup.sh (Linux)
```

## Reproduce Command
```bash
python rank.py --candidates ./candidates.jsonl --out ./outputs/mohd_ibadullah.csv
```
