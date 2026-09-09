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
cross-encoder/ettin-reranker-17m-v1 on top 150 — 60% feature + 40% CE blend
         ↓
[Stage 5] Profile-Specific Reasoning (offline, rule-based)
Rank-aware summaries citing skills, company, Redrob signals — no API calls
         ↓
[Stage 6] CSV Output
Top 100 ranked candidates — validator PASS, monotonic scores
         ↓
Output: sample_ranking_output.csv — ~80–150 seconds warm (models + embeddings cached)
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
| Total Runtime (100K, warm) | **~80–150 seconds** (~82.7s internal / ~149.9s wall) |
| First run (model download + no cache) | up to ~8 min — run `setup.ps1` once before judging |
| Peak Memory | **~2–3 GB** (under 16 GB budget) |
| Stage 5 Cross-Encoder Rerank | ~6 seconds |
| Honeypot Rate in Top 100 | 0.0% |
| CSV Validation | PASS |

## System Validation (TalentLens AI vs BM25 Baseline)
Adversarial filtering and candidate discovery comparison against an unweighted lexical baseline:

* **Top 100 Honeypots:** 0.0% (TalentLens AI) vs 6.0% (BM25 baseline)
* **Top 1000 Honeypot Filtering:** 362 decoy profiles (36.2% of BM25 pool) intercepted and filtered out before final ranking
* **Ground-Truth Ranking Quality:** Final Precision, Recall, and NDCG depend on private competition gold-standard relevance labels

## One-Time Setup (before Stage 3 reproduction)
```bash
pip install -r requirements.txt
python src/download_models.py
python src/precompute_embeddings.py
# Or: .\setup.ps1  (Windows) / ./setup.sh (Linux)
```

## Reproduce Command
```bash
python rank.py --candidates ./candidates.jsonl --out ./outputs/sample_ranking_output.csv
```
