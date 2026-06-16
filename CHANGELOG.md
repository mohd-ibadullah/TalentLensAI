# TalentLens AI — Development Changelog

## v1.3.0 — Recruiter Explainability & Fairness (Final)
- Added Gemini 2.5 Flash powered Interview Question Generator in Streamlit
- Added Fairness & Responsible AI documentation
- Advanced Signal Envelopes: notice period bonus, inactivity penalty
- Runtime: ~139s on 100K candidates

## v1.2.0 — Pipeline Speed Optimization  
- Reduced execution time from 304s to ~139s (54% speedup)
- Single-pass JSONL streaming, batch size 128, max_length 160
- BM25 filter tuned to 1000 candidates for optimal speed/quality balance
- 97% ranking overlap vs previous version

## v1.1.0 — Core Scoring Engine
- Semantic embedding scorer using BAAI/bge-base-en-v1.5
- Weighted feature scoring: semantic(35%) + skills(30%) + title(15%) + signals(10%) - trap(40%)
- LLM reranker with dynamic non-templated reasoning generator
- Validated: 0 honeypots in top 100, CSV passes validate_submission.py

## v1.0.0 — Foundation
- BM25 coarse filter: 100K → 1000 candidates in seconds
- Honeypot trap detector: career-title consistency checks
- JD parser with excluded skills protection (no Computer Vision false matches)
- Data loader with memory-efficient JSONL streaming
