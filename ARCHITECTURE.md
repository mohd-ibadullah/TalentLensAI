# TalentLens AI — System Architecture

## Pipeline Overview
```
Job Description (text/JSON)
         ↓
[Stage 1] BM25 Lexical Filter
100,000 candidates → top 1,000 (milliseconds)
         ↓
[Stage 2] Honeypot Trap Detector
Flags career-title mismatches, keyword stuffing, template profiles
         ↓  
[Stage 3] Semantic Embedding Scorer
BAAI/bge-base-en-v1.5 local model, cosine similarity
         ↓
[Stage 4] Feature Scoring Engine
Semantic(35%) + Skills(30%) + Title/YoE(15%) + Signals(10%) - Trap Penalty(40%)
         ↓
[Stage 5] Cross-Encoder Reranker
cross-encoder/ms-marco-MiniLM-L6-v2 pairwise relevance scoring
Blended: 60% feature score + 40% cross-encoder score
         ↓
[Stage 6] Dynamic Reasoning Generator
Non-templated, rank-aware, concern-honest candidate summaries
         ↓
Output: Top 100 ranked candidates (CSV) — ~151 seconds total
```

## Key Design Decisions
- Local model only (no API calls during ranking) — CPU-safe, offline-reproducible
- Single-pass streaming — handles 487MB JSONL without memory overflow
- Honeypot detection first — prevents decoys from reaching expensive embedding stage
- Cross-encoder reranker on top 150 — pairwise precision boost without runtime blowup
- -1 sentinel values treated as neutral (not penalized) across all signal fields
- Excluded skills (Computer Vision, Robotics) explicitly not matched to JD

## Performance Benchmarks
| Metric | Value |
|--------|-------|
| Total Runtime (100K candidates) | ~151 seconds |
| Stage 1 BM25 Filter | ~8 seconds |
| Stage 2-4 Embedding + Scoring | ~130 seconds |
| Stage 5 Cross-Encoder Rerank | ~4 seconds |
| Stage 6 Reasoning + Output | ~7 seconds |
| Honeypot Rate in Top 100 | 0.0% |
| CSV Validation | PASS |
