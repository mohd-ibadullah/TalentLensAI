# TalentLens AI — Complete Technical Documentation

> **Version:** 1.0 (September 2026)
> **Audience:** An engineer with zero prior context on this repository.
> **Everything below is derived from the actual code, docs, and paper in this repo.** Where information does not exist in the repo, the section explicitly says "Not yet defined".

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation / Background](#2-motivation--background)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Core Modules / Components](#5-core-modules--components)
6. [Data](#6-data)
7. [Models](#7-models)
8. [Algorithms / Key Logic](#8-algorithms--key-logic)
9. [File/Folder Structure](#9-filefolder-structure)
10. [APIs / Interfaces](#10-apis--interfaces)
11. [Setup & Installation](#11-setup--installation)
12. [Current Status](#12-current-status)
13. [Known Issues / Limitations](#13-known-issues--limitations)
14. [Next Steps / Roadmap](#14-next-steps--roadmap)
15. [Team & Roles](#15-team--roles)
16. [Constraints](#16-constraints)

---

## 1. Project Overview

**TalentLens AI** is a candidate-ranking system that takes a large pool of candidate
profiles (tested at 100,000 profiles) and a job description (JD), and produces a ranked
shortlist of the best-matching candidates with per-candidate reasoning.

**What it does (end to end):**

1. **Parses a job description** (structured JSON or free text) into role title, required /
   nice-to-have skills, minimum years of experience, seniority level, and domain keywords.
2. **Hybrid retrieval:** fuses sparse lexical search (**BM25**, via `rank_bm25`) with dense
   semantic search (**BAAI/bge-base-en-v1.5** sentence embeddings) over the candidate corpus.
3. **Honeypot screening:** a rule-based decoy detector flags keyword-stuffed and
   template/boilerplate profiles and hard-vetoes them (score → 0).
4. **Feature scoring:** weighted combination of semantic similarity, skill match, title /
   seniority fit, and recruiter-platform engagement signals, plus hard gates
   (YoE gate, disqualifier rules).
5. **Cross-encoder reranking:** the top ~150 candidates are rescored pairwise with
   **ettin-reranker-17m-v1** and blended (60/40) with the feature score.
6. **Reasoning generation + CSV output:** rule-based (or optionally LLM-based) recruiter-style
   rationale per candidate, written to a submission CSV.

**Problem it solves:** recruiters drowning in tens of thousands of resumes need a fast,
CPU-only, measurable, and **fair** ranking system. A central, verified property of this
project is **name blindness**: candidate names never influence the ranking (validated by an
800/800 name-swap invariance test), and a research paper accompanying the repo quantifies
where identity (name) leakage occurs in typical retrieval pipelines and how to remove it at
zero cost.

**Who it's for:**

- Recruiters / hiring teams doing high-volume screening (via the Streamlit dashboard).
- ML competition judges (the repo was built for the **India Runs on Data & AI Challenge**,
  Redrob-style candidate-ranking task with a strict CSV submission schema).
- IR/fairness researchers (the repo backs a CEUR-WS-style paper for TalentCLEF 2026).

---

## 2. Motivation / Background

Two motivations drive the project:

### 2.1 The engineering/competition motivation
The India Runs on Data & AI Challenge (sponsored by Redrob — hence the `redrob_signals`
field in candidate profiles) required ranking 100,000 candidate profiles against a JD on
CPU, with honeypot ("trap") profiles seeded into the pool to catch naive systems, and a
fixed submission format (`candidate_id, rank, score, reasoning`). The system must also
produce human-readable recruiter rationale for Stage-4 manual review.

### 2.2 The research motivation — identity leakage in resume retrieval
The repo backs a measured study:

> **"Where Does the Name Leak? A Stage-Wise Decomposition of Identity Sensitivity in
> Resume Retrieval"** — Mohd Ibadullah (Keshav Memorial Institute of Technology, JNTUH).

Key findings (from `paper/PAPER_DRAFT.tex`):

- On **TalentCLEF 2026 Task A** (CC-BY-4.0, DOI 10.5281/zenodo.17625261; 472 CVs per split,
  English + Spanish, official human relevance judgments):
  - **BM25 is nearly name-blind** — Rank-Biased Overlap (RBO) ≈ 0.99 under name swaps.
  - **Dense retrieval (bge-base-en-v1.5) reorders candidates substantially** when only the
    name changes: RBO 0.669 (English), 0.639 (Spanish).
  - Hybrid fusion and cross-encoder reranking fall in between: RBO 0.773 and 0.899.
  - **Removing the identity header** (3 lines of preprocessing) raises RBO to exactly 1.000
    in both languages at no measurable accuracy cost in English; in Spanish it even improves
    dense retrieval average precision by +0.094 (95% CI +0.041 to +0.146, corrected p=0.0003).
  - For context: the best system among 113 teams in the shared task reached RBO 0.9904 on
    the held-out test split using a large-parameter ensemble.
- Related work cited: Wilson & Caliskan (embedding bias in hiring), Rao et al. 2025
  ("Invisible" — cross-cultural LLM hiring bias), Wilson et al. 2025 (human+AI bias propagation).

**Consequence for the product:** the production pipeline (`src/`) deliberately builds all
candidate text representations (`src/candidate_text.py`, `src/bm25_filter.py`) **without
reading the candidate name field**. This is not just policy — it is enforced by tests
(`tests/test_name_blindness.py`, `tests/test_text_builder_name_blindness.py`).

**Honesty commitment:** the README explicitly discloses that earlier drafts claimed a
"150% lift" from a circular evaluation; those numbers were wrong and were removed. All
remaining metrics are script-measured, not estimated.

---

## 3. Architecture

### 3.1 High-level flow

```
                          ┌──────────────────────────────┐
                          │  Job Description input        │
                          │  (JSON config or free text)   │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │  Stage 0: JD Parser           │  src/jd_parser.py
                          │  role_title, skills, min YoE, │
                          │  seniority, domain keywords   │
                          └──────────────┬───────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                ▼                                │
┌───────▼────────┐            ┌──────────────────────┐          ┌────────▼─────────┐
│ candidates.jsonl│ stream    │ Stage 1: HYBRID      │          │ data/            │
│ (100K profiles) ├──────────►│ RETRIEVAL            │◄─────────┤ candidate_       │
└────────┬────────┘ pass 1     │                      │  precomp │ embeddings.npy   │
         │                     │ • BM25 lexical top-1000       │  100K×768        │
         │                     │   (rank_bm25 BM25Okapi)       │  candidate_ids   │
         │                     │ • Dense vector top-1000       │  .json           │
         │                     │   (bge-base-en-v1.5,          │  embeddings_meta │
         │                     │    dot product, CLS pool)     │  .json           │
         │                     │ • Union of both recalls       └────────┬─────────┘
         │                     └──────────┬───────────────────┘          │
         │ stream pass 2 (fetch full      │ ~≤2000 unique candidates     │
         │ profiles only for needed IDs)  ▼                              │
         │               ┌──────────────────────────────┐                │
         └──────────────►│ Stage 2: HONEYPOT DETECTOR   │                │
                         │ detect_trap() → score [0,1]  │                │
                         │ decoy/boilerplate/stuffing   │                │
                         └──────────────┬───────────────┘                │
                                        ▼                                │
                         ┌──────────────────────────────┐                │
                         │ Stage 3: FEATURE SCORER       │                │
                         │ weighted sum (0.40 sem, 0.20  │                │
                         │ skills, 0.20 title, 0.10      │                │
                         │ signals) + career bonus +     │                │
                         │ disqualifiers + hard vetoes   │                │
                         └──────────────┬───────────────┘                │
                                        ▼                                │
                         ┌──────────────────────────────┐                │
                         │ Stage 4: sort, take top 150   │                │
                         └──────────────┬───────────────┘                │
                                        ▼                                │
                         ┌──────────────────────────────┐                │
                         │ Stage 5: CROSS-ENCODER        │                │
                         │ ettin-reranker-17m-v1         │                │
                         │ blend = 0.6·feature + 0.4·CE  │                │
                         └──────────────┬───────────────┘                │
                                        ▼                                │
                         ┌──────────────────────────────┐                │
                         │ Stage 6: REASONING GENERATOR  │                │
                         │ rule-based (default) or LLM   │                │
                         │ (Gemini/OpenAI/Groq optional) │                │
                         └──────────────┬───────────────┘                │
                                        ▼                                │
                         ┌──────────────────────────────┐                │
                         │ OUTPUT CSV                    │                │
                         │ candidate_id, rank, score,    │                │
                         │ reasoning  (score ≤ 0.994)    │                │
                         └──────────────────────────────┘
```

### 3.2 Two run modes

| Mode | Entry point | Dataset | Behavior |
| --- | --- | --- | --- |
| **CLI / production** | `python rank.py --candidates ./candidates.jsonl --out ./outputs/participant_id.csv` | 100K JSONL | Two-pass streaming (never loads all 100K full profiles into RAM), precomputed embedding cache with fingerprint verification, preflight checks. |
| **Dashboard / interactive** | `streamlit run app/streamlit_app.py` (or `python run_app.py` for cloud deployment) | 50-candidate sample **or** full 100K | Same pipeline stages; full mode streams + caches via `st.cache_resource`; adds LLM interview-question generation per candidate. |

### 3.3 Caching layer (critical detail)

`data/candidate_embeddings.npy` (100,000 × 768 float32, ~300 MB, **not committed to git**)
plus `data/candidate_ids.json` are the embedding cache. `src/pipeline.py` refuses the cache
unless **both** fingerprints verify:

1. **ID fingerprint** — SHA-256 of the candidate ID list must match the current JSONL.
2. **Text fingerprint** — SHA-256 over (model name, max_seq_length, n_candidates, plus the
   embedding text of every 500th candidate) recomputed from the live data and compared to
   `data/embeddings_meta.json`. If `candidate_text.py` logic changed, the cache is rejected
   and embeddings are recomputed on the fly (BM25-only fallback with a warning).

This prevents the classic "stale vectors silently poison the ranking" failure.

### 3.4 Memory strategy

- Corpus documents for BM25 are built by streaming; only ~top-1000 (BM25) ∪ top-1000 (dense)
  full profiles are loaded for deep scoring.
- Embedding matrix is loaded with `np.load(..., mmap_mode="r")` — ~300 MB on disk, paged in
  on demand, never duplicated in RAM.
- `EmbeddingScorer.id_to_index` maps candidate_id → row for O(1) similarity lookup.

---

## 4. Tech Stack

### 4.1 Languages & runtime
- **Python** (3.10+ syntax used, e.g. `list[str]`, `dict | None`) — the only language.

### 4.2 Python libraries (from `requirements.txt`, with minimum versions)

| Library | Min version | Used for |
| --- | --- | --- |
| pandas | ≥2.0.0 | CSV I/O, DataFrame output |
| numpy | ≥1.24.0 | embedding matrix, math, clipping |
| scikit-learn | ≥1.0.0 | listed as dependency (general ML utilities) |
| rapidfuzz | ≥3.0.0 | fuzzy skill matching (`fuzz.token_sort_ratio`) |
| streamlit | ≥1.20.0 | interactive dashboard |
| transformers | ≥4.30.0 | loading BGE embedding model |
| torch | ≥2.0.0 | tensor inference (CPU) |
| PyYAML | ≥6.0 | YAML parsing (paper metadata) |
| rank-bm25 | ≥0.2.2 | `BM25Okapi` lexical retrieval |
| sentence-transformers | ≥2.2.0 | `CrossEncoder` reranking |
| python-dotenv | ≥1.0.0 | .env loading support |
| pytest | ≥7.0.0 | 177 tests |
| openpyxl | ≥3.1.0 | Excel export helper (`src/csv_to_xlsx.py`) |
| requests | ≥2.28.0 | LLM API calls (Gemini / Groq) |

### 4.3 ML models

| Model | Type | Provider/HF ID | Role |
| --- | --- | --- | --- |
| bge-base-en-v1.5 | Bi-encoder, 768-dim embeddings | `BAAI/bge-base-en-v1.5` | Dense semantic retrieval + semantic similarity score. Chosen over MiniLM-L6-v2 (384-dim) for retrieval quality while remaining CPU-fast. Uses `[CLS]` pooling, L2 normalization, and the BGE query instruction prefix `"Represent this sentence for searching relevant passages: "` for queries. |
| ettin-reranker-17m-v1 | Cross-encoder (17.6M params) | `cross-encoder/ettin-reranker-17m-v1` | Reranking of top ~150 candidates. Chosen because it is "strictly dominant": +0.049 NDCG@10 over `ms-marco-MiniLM-L6-v2`, 1.86× faster on CPU (i7-13700K), 5.2M fewer parameters. `max_length=256`. |
| gemini-2.5-flash / gpt-oss-120b (Groq) / gemini-1.5-flash / gpt-4o-mini | External LLM APIs | Google / Groq / OpenAI | **Optional only.** Interview-question generation (dashboard) and LLM-based reasoning (off by default). Ranking itself never requires an API key. |

### 4.4 Tools & infrastructure
- **Streamlit Cloud** — public live demo: `https://talentlensai-nxrk7zxjmaxvnwnubyvz7n.streamlit.app/` (sample dataset only; cloud guard via `STREAMLIT_RUNTIME_ENV`).
- **pytest** — test suite (177 passing tests reported).
- **LaTeX (CEUR-WS `ceurart` class)** — research paper in `paper/`.
- **Streamlit/Starlette gzip monkeypatch** — `run_app.py` and the top of `app/streamlit_app.py` patch `starlette.middleware.gzip` for headless/cloud deployment compatibility.
- License: **MIT** for code; benchmark data remains under the TalentCLEF license with attribution.

---

## 5. Core Modules / Components

All pipeline modules live in `src/` and are imported flat (`sys.path.append(dirname(__file__))`).

| Module | Responsibility |
| --- | --- |
| **`src/pipeline.py`** | `run_ranking_pipeline(candidates_path, jd_input, out_csv_path, top_n=100, use_llm=False, weights=None)` — orchestrates all 6 stages end-to-end. Handles both in-memory sample mode and streaming 100K production mode, embedding cache verification, and final CSV writing with schema `candidate_id, rank, score, reasoning`. |
| **`src/jd_parser.py`** | `parse_job_description(jd_input)` — accepts a structured dict or free text; sanitizes fields with defaults (role "Senior AI Engineer", 5.0 YoE, "Senior"); for free text: regex-extracts YoE (`(\d+(\.\d+)?)\s*\+?\s*(?:years?|yrs?)`), seniority keywords, role title (`role|title|position:` or known title list), and skills by matching a 40-entry `COMMON_SKILLS_TAXONOMY`, split into required vs nice-to-have by section headers (must-have/required/essential vs nice-to-have/preferred/plus/optional). Derives domain keywords. |
| **`src/bm25_filter.py`** | `build_candidate_document(cand)` builds a lexical document from current_title + summary + all skill names + career titles + first 200 chars of each career description (**no candidate name**). `BM25Filter` wraps `BM25Okapi`: constructor from candidate list, or `from_corpus(corpus_docs)` classmethod for the memory-efficient streaming path; `get_top_indices(parsed_jd, top_n)`, `filter_candidates(parsed_jd, top_n)`; `_build_query` tokenizes role title + required + nice-to-have + domain keywords. Tokenizer: `re.findall(r'\b\w+\b', text.lower())`. |
| **`src/embedding_scorer.py`** | `EmbeddingScorer` — loads `BAAI/bge-base-en-v1.5` on CPU. `get_embeddings(texts, is_query)` batches (128), truncates at 160 tokens, CLS-pools, L2-normalizes. `load_precomputed_embeddings(npy, json)` mmaps the 100K×768 matrix. `search_similar_candidates(jd_text, top_n)` = dot product against whole matrix + top-k argsort. `get_candidate_similarity_by_id(cid, jd_vec)` per-candidate lookup with neutral 0.5 fallback. `compute_similarity(jd_text, cand_texts)` for on-the-fly encoding. |
| **`src/honeypot_detector.py`** | `detect_trap(cand) -> (trap_score, reason)` — rule-based decoy scoring (details in §8.2). Also exposes `classify_title()` (title → category via keyword map) and `check_boilerplate_description()` (matches known decoy templates like "business analyst at a consulting firm"). |
| **`src/feature_scorer.py`** | `calculate_candidate_score(cand, semantic_similarity, trap_score, parsed_jd, weights)` — the heart of scoring: weighted component sum + career bonus + disqualifier penalties + multiplicative veto factors (details in §8.3). Also `match_skill` (exact/substring/fuzzy via RapidFuzz), `compute_skill_match_score`, `compute_title_seniority_match`, `compute_signal_bonus`. |
| **`src/cross_encoder_reranker.py`** | `CrossEncoderReranker.rerank(jd_text, candidates, blend_weight=0.4, min_yoe)` — builds concise pair text (title, YoE, 200-char summary, top-12 skills), batch-predicts CE scores, min-max normalizes to [0,100], blends 60/40 with `_final_score`, light −10 nudge for YoE-deficit borderline cases, re-sorts and reassigns `_rank`. |
| **`src/llm_reranker.py`** | `rerank_top_candidates(top_candidates, parsed_jd, use_llm=False)` — final ranking/capping (score clamped to [0, 99.4]), tie-broken sort by `(-round(score/100, 4), candidate_id)` per the submission spec, `_rank` assignment, and reasoning generation: `generate_rule_based_reasoning()` produces a fact-grounded recruiter sentence (tone banded: Top-tier ≥85, Strong ≥70, Moderate ≥50, Weak <50) citing matched JD skills, Redrob signals, career history, and concerns. Optional LLM paths (Gemini 1.5 Flash via `google.generativeai`, OpenAI gpt-4o-mini) when keys exist and `use_llm=True`. |
| **`src/candidate_text.py`** | `build_candidate_embedding_text(cand)` — the single shared text builder for dense embeddings: `Title / Headline / Skills (top 15) / Career titles (top 5) / Summary`. **Must match `precompute_embeddings.py` exactly; contains no name field.** |
| **`src/data_loader.py`** | `load_sample_candidates(path)` (JSON array) and `stream_candidates(path)` (JSONL generator). Both run the corpus integrity guard on every load. |
| **`src/data_integrity.py`** | `check_corpus(path, min_populated_fraction=0.95)` — SHA-256 the file, count JSON-parse errors, fraction of records with ≥1 skill, median line bytes; flags shell-record corpora (median < 500 bytes). `print_corpus_summary` prints one-line stats + loud warning. Measured result: 100,000 profiles, 100% populated. |
| **`src/precompute_embeddings.py`** | Offline script that streams all candidates, builds embedding texts, encodes with BGE, writes `data/candidate_embeddings.npy` + `data/candidate_ids.json` + `data/embeddings_meta.json` (with text fingerprint). Locates `candidates.jsonl` via multiple known paths. |
| **`src/download_models.py`** | Pre-downloads/caches bge-base-en-v1.5, ettin-reranker-17m-v1 (and ettin-reranker-17m) so the ranking pipeline can run fully offline (Stage-3 sandbox verification requirement). |
| **`src/preflight.py`** | `run_preflight(...)` and `ensure_embeddings_exist(...)` — model-cache preflight checks before ranking. |
| **`src/run_pipeline_full.py`** | Production CLI: argparse (`--candidates`, `--jd`, `--out`, `--validate`, `--skip-preflight`, `--check-pool`, `--setup`, `--allow-bm25-only`), fallback path search for candidates.jsonl, optional post-run validation + candidate-pool ID check. |
| **`src/evaluate.py`, `src/explore.py`, `src/test_honeypot.py`, `src/test_pipeline.py`** | Legacy/dev evaluation and exploratory scripts (predecessors of `scripts/` and `tests/`). |
| **`src/csv_to_xlsx.py`** | CSV → XLSX export helper. |
| **`app/streamlit_app.py`** | The full dashboard (detailed in §10.2). Adds display-only cleanup (salary min/max swap fix, education dedup, summary rewording), quality-threshold grouping, downloadable CSV, per-candidate LLM interview questions. |
| **`run_app.py`** | Headless cloud launcher: monkeypatches Starlette gzip, launches `streamlit run app/streamlit_app.py` on port 8501 headless. |

---

## 6. Data

### 6.1 Primary dataset — `candidates.jsonl` (100,000 profiles)

- **Source:** the *India Runs on Data & AI Challenge* bundle (`[PUB] India_runs_data_and_ai_challenge`), sponsored by Redrob. The full 100K file is **not committed**; the repo ships `data/candidates.jsonl` reference and `data/sample_candidates.json` (50-candidate dev sample).
- **Format:** one JSON object per line. Record schema (fields consumed by the code):

```
{
  "candidate_id": str,
  "profile": {
    "anonymized_name": str,        # display only — NEVER used in scoring
    "current_title": str,
    "current_company": str,
    "current_industry": str,
    "headline": str,
    "summary": str,
    "years_of_experience": float,
    "location": str,
    "country": str
  },
  "skills": [ { "name": str, "proficiency": "beginner|intermediate|advanced|expert",
                "endorsements": int, "duration_months": int } ],
  "career_history": [ { "title": str, "company": str, "description": str,
                        "duration_months": int, "start_date": "YYYY-..." } ],
  "education": [ { "degree": str, "field_of_study": str, "institution": str,
                   "start_year": int, "end_year": int, "tier": str } ],
  "redrob_signals": {
    "profile_completeness_score": 0–100 | -1,
    "recruiter_response_rate": 0–1 | -1,
    "interview_completion_rate": 0–1 | -1,
    "github_activity_score": 0–100 | -1,
    "open_to_work_flag": bool,
    "notice_period_days": int | -1,
    "last_active_date": "YYYY-MM-DD",
    "expected_salary_range_inr_lpa": { "min": float, "max": float }
  }
}
```

- **Integrity:** every load runs `data_integrity.check_corpus` (SHA-256, populated fraction ≥ 0.95, median line ≥ 500 bytes, JSON-parse check). Measured: 100,000/100,000 populated.

### 6.2 Precomputed artifacts (generated, not committed)

| File | Contents |
| --- | --- |
| `data/candidate_embeddings.npy` | 100,000 × 768 float32, L2-normalized BGE embeddings (~300 MB) |
| `data/candidate_ids.json` | candidate ID list whose order matches the matrix rows |
| `data/embeddings_meta.json` | `{model_name, max_seq_length, n_candidates, text_fingerprint}` for cache validation |

### 6.3 Research benchmark — TalentCLEF 2026 Task A

- Used **only for the paper**, not the product pipeline.
- 472 CVs per split, English and Spanish, official human relevance judgments (qrels).
- License CC-BY-4.0, DOI 10.5281/zenodo.17625261.

### 6.4 Preprocessing steps

1. Corpus integrity guard on every load.
2. **Name stripping by omission:** text builders (`candidate_text.py`, `bm25_filter.py`) simply never include the name field — this is the zero-cost fairness intervention validated by the paper.
3. Embedding text: top 15 skills, top 5 career titles, 160-token max truncation at encode time.
4. BM25 document: title + summary + all skills + career titles + first 200 chars per description.
5. Display-only fixes in the dashboard (reversed salary ranges, education dedup via degree-alias map, title backtick stripping) that never mutate scoring inputs.

---

## 7. Models

### 7.1 BAAI/bge-base-en-v1.5 (bi-encoder)

- **Why:** best CPU-viable retrieval quality; README notes 768-dim BGE outperforms MiniLM-L6-v2 (384-dim) while staying fast; paper uses it for the dense stage of the identity-leak decomposition.
- **Training/fine-tuning:** none — used off-the-shelf. No training code exists in the repo (Not yet defined — no fine-tuning pipeline).
- **Usage details:**
  - Loaded via `transformers.AutoModel` on CPU, eval mode.
  - `[CLS]` pooling (BGE-recommended), L2 normalization, batch 128, `max_length=160`.
  - **Asymmetric query:** queries get the BGE instruction prefix; candidate passages do not.
- **Inputs:** JD embedding text (`role_title + required skills + nice-to-have + domain + seniority`) and candidate text from `build_candidate_embedding_text`.
- **Outputs:** 768-dim vectors → cosine (dot) similarities.
- **Pipeline role:** Stage-1 dense recall (top 1000) and the `semantic_similarity` feature (weight 0.40).

### 7.2 cross-encoder/ettin-reranker-17m-v1 (cross-encoder)

- **Why:** strictly dominant swap for `ms-marco-MiniLM-L6-v2` — +0.049 NDCG@10, 1.86× faster on CPU, 17.6M vs 22.7M params (docstring-measured on i7-13700K).
- **Training/fine-tuning:** none — off-the-shelf via `sentence_transformers.CrossEncoder`, `max_length=256`.
- **Inputs:** pairs `(jd_embedding_text, candidate_text)` where candidate text = `"{title} ({yoe} years). {summary[:200]}. Skills: {top 12 skills}"`.
- **Outputs:** raw relevance logits → min-max normalized to [0, 100] within the rerank batch → blended `0.6·feature_score + 0.4·CE_score`.
- **Pipeline role:** Stage-5 precision boost on the top ~150 candidates.

### 7.3 Optional LLMs (never required)

- **Dashboard interview questions:** Gemini 2.5 Flash (or Groq-hosted `openai/gpt-oss-120b` when the key starts with `gsk_`), temperature 0.5, max 800 tokens, 3 gap-probing questions, single retry on 429/timeout/empty.
- **Reasoning generation (off by default):** Gemini 1.5 Flash or gpt-4o-mini, ≤25–30 word recruiter sentences; automatic fallback to the rule-based generator on any failure.

### 7.4 Ensemble / pipeline logic

There is no trained ensemble. The "ensemble" is a **cascade**: cheap lexical+dense recall → veto gates → feature scoring → expensive pairwise rerank of survivors only. Blend weights (0.4 CE, weights in §8.3) are fixed defaults, tunable live in the dashboard sidebar.

---

## 8. Algorithms / Key Logic

### 8.1 Hybrid retrieval

1. Tokenize corpus documents and JD query (`\b\w+\b`, lowercased).
2. `BM25Okapi.get_scores(query)` over all 100K docs → top **1000** indices.
3. Dense: embed JD (query-prefixed), dot product against the 100K×768 matrix → top **1000** candidate IDs.
4. Union both ID sets; stream the JSONL a second time loading only the needed full profiles (~≤2000).
5. BM25-only fallback if the embedding cache is missing/rejected (or `--allow-bm25-only`).

### 8.2 Honeypot detection (`detect_trap`) — additive rule scoring, capped to [0, 1]

Signals (weights as coded):

| Signal | Points | Trigger |
| --- | --- | --- |
| Impossible timeline | +1.0 each | Career role `start_date` year < company "founded in YYYY" year found in its description |
| Decoy summary template | +0.4 | Summary contains "marketing manager" while title isn't Marketing; or "curious about how ai tools could augment my work"; or "experimented with chatgpt" |
| AI-skill stuffing on non-AI title | +0.25 (≥1 skill) or +0.1·count up to 0.4 (≥3) | Title classifies as HR/Accounting/Support/etc. (not AI/ML, SWE, Other, Unknown) while listing AI/ML skills from a 27-skill set |
| Expert skill, zero duration | +0.35 (once) | Any skill with proficiency expert/advanced and `duration_months == 0` |
| Boilerplate career descriptions | +0.35 (once, if ≥2 roles) | ≥2 roles match the known decoy-template map |
| Title/description category mismatch ratio | +0.3 × ratio | Fraction of roles whose boilerplate description category ≠ title category |

A trap_score **≥ 0.40 vetoes the candidate completely** (multiplicative factor 0 → score exactly 0.0).

### 8.3 Feature scoring (`calculate_candidate_score`) — full decision logic

**Positive weighted sum** (defaults; UI-tunable):

```
scaled_positive = (
    0.40·semantic_similarity      # BGE cosine
  + 0.20·skill_match_score
  + 0.20·title_seniority_match
  + 0.10·signal_bonus
) / sum(positive weights) × 100
```

- **skill_match_score:** for each candidate skill, fuzzy-match (`RapidFuzz.token_sort_ratio ≥ 85`, or exact 1.0 / substring 0.9) against required (+1.0) and nice-to-have (+0.5) skills; multiplied by proficiency multiplier (beginner 0.6 / intermediate 0.8 / advanced 1.0 / expert 1.2) and endorsement bonus (up to +20% at 50 endorsements); normalized by required-count, capped 0.994. Empty required list → 1.0.
- **title_seniority_match** (hard cap 0.97, never 1.0): seniority 40% (senior-keyword → 1.0, mid-keyword → 0.72, else 0.40), domain 30% (AI keywords → 1.0, negative domains (mechanical, sales, HR, …) → 0.05, generic software → 0.50, else 0.30), YoE 30% (≥ min: `0.75 + surplus·0.025` capped 1.0; below: proportional `yoe/min·0.75`).
- **signal_bonus:** mean of available normalized Redrob signals (completeness/100, response rate, interview completion, GitHub/100, open-to-work 0.8/0.2; neutral 0.5 if none) + modifiers: +0.05 notice ≤30d, −0.05 response <0.15, −0.03 inactive >180 days (vs ref date 2026-06-16). Clipped [0,1].

**Career relevance bonus:** +3 per career-history role/description matching keywords (ranking, retrieval, recommendation, search, embedding, recruiter, product company), capped +10.

**Disqualifier rules** (each multiplies the score):

| Rule | Factor | Condition |
| --- | --- | --- |
| Consulting-heavy career | ×0.50 | >60% of total career months at TCS/Infosys/Wipro/Cognizant/Accenture/HCL/Tech Mahindra/Capgemini/L&T. *Disclosed policy choice, documented as unfair to one employer group in the paper (Fisher exact p=0.0046 at top-20).* |
| Pure CV/speech/robotics (no NLP/IR) | ×0.50 | CV keywords present, no NLP/IR keywords |
| Marketing/HR title + AI stuffing | ×0.0 | Absolute rejection |
| LangChain-only, no production ML & no ML titles | ×0.50 | LangChain present; no PyTorch/TF/sklearn/MLOps/K8s/Docker/AWS… |
| Non-tech engineer, no AI/ML | ×0.0 | Mechanical/civil/electrical/chemical/structural/environmental/industrial engineer with no NLP/IR and no production ML |

**Multiplicative vetoes:**

```
trap_factor  = 0.0 if trap_score ≥ 0.40 else 1.0
yoe_factor   = 0.0 if years_of_experience < JD min (default 5.0) else 1.0   # hard gate
disq_factor  = product of rule factors above

base_score   = scaled_positive + career_bonus
final_score  = base_score × trap_factor × yoe_factor × disq_factor + behavioral_adjustment
final_score  = clamp(final_score, 0.0, 99.4)   # never a flat 100.0
```

**Behavioral adjustments:** −10.0 inactive >180 days, −5.0 notice >60 days, +5.0 open-to-work AND response ≥0.70. (Low response rate is *not* double-penalized here.)

### 8.4 Cross-encoder blend & YoE nudge

`blended = 0.6·original_feature_score + 0.4·minmax_normalized_CE_score`, minus 10.0 if `yoe < min_yoe` (borderline nudge; the hard gate already applied upstream). Re-sort descending with `candidate_id` ascending tie-break; ranks reassigned 1..N.

### 8.5 Final ranking & tie-break (submission spec)

Scores capped at 99.4; sorted by `(-round(score/100, 4), candidate_id)` so CSV order matches model order under equal scores; output columns exactly `candidate_id, rank, score (0–1 float, 4 dp), reasoning`.

### 8.6 Name-blindness guarantee

- No scoring/ranking code path reads `profile.anonymized_name` or any name field.
- The **only** exception (documented, not hidden): the consulting-career rule reads **employer** names.
- Verified: 800/800 name swaps produce byte-identical rankings (`scripts/measure_name_blindness.py`); dedicated unit tests assert the text builders never emit names.

---

## 9. File/Folder Structure

```
TalentLensAI/
├── rank.py                      # Official CLI entry point (alias for src.run_pipeline_full.main)
├── run_app.py                   # Headless cloud launcher for the Streamlit app (Starlette gzip patch)
├── requirements.txt             # Python dependencies (14 pinned-minimum libs)
├── setup.sh / setup.ps1         # One-command setup for Linux/Mac and Windows (models + embeddings)
├── .env.example                 # Template: optional GEMINI_API_KEY / api_key (Groq) — NOT needed for ranking
├── .gitignore                   # Excludes .env, embeddings cache, caches, junk
├── .mailmap                     # Git identity canonicalization
├── LICENSE                      # MIT
├── README.md                    # Product overview, measured results, quick start, live demo link
├── ARCHITECTURE.md              # Architecture notes
├── CHANGELOG.md                 # Release history
├── Project_Abstract.md          # Short abstract of the project/study
├── INTERVIEW_PREP_GUIDE.md      # Author's interview-prep notes about this project
├── TECHNICAL_DOCUMENTATION.md   # This file
├── check_tex.py                 # LaTeX source sanity checker for the paper
├── _qa_audit.py                 # Root-level QA audit helper
│
├── src/                         # Core pipeline package (all ranking logic)
│   ├── pipeline.py              # Orchestrator: 6-stage run_ranking_pipeline()
│   ├── run_pipeline_full.py     # Production CLI with argparse, preflight, validation
│   ├── jd_parser.py             # JD parsing (dict or free text → structured requirements)
│   ├── bm25_filter.py           # BM25Okapi lexical retrieval + candidate document builder
│   ├── embedding_scorer.py      # BGE encoder, cache loader, dense search
│   ├── cross_encoder_reranker.py# ettin-reranker-17m-v1 pairwise reranking + blend
│   ├── feature_scorer.py        # Weighted scoring, skill/title/signal components, vetoes
│   ├── honeypot_detector.py     # Rule-based decoy/trap detection
│   ├── llm_reranker.py          # Final ranking, capping, tie-break, reasoning generation
│   ├── llm_reasoning_generator.py # LLM reasoning batch generator (dev)
│   ├── candidate_text.py        # Shared name-free embedding text builder
│   ├── data_loader.py           # Sample loader + JSONL streamer (runs integrity guard)
│   ├── data_integrity.py        # Corpus SHA-256/population/size validator
│   ├── precompute_embeddings.py # Offline embedding precomputation + fingerprints
│   ├── download_models.py       # Model cache pre-download for offline runs
│   ├── preflight.py             # Pre-run environment checks
│   ├── evaluate.py / explore.py # Legacy evaluation/exploration
│   ├── test_honeypot.py / test_pipeline.py # Legacy dev tests (canonical tests are in tests/)
│   └── csv_to_xlsx.py           # CSV → Excel converter
│
├── app/
│   └── streamlit_app.py         # Full dashboard: dataset switcher, weight sliders, ranking UI,
│                                #   score cards, CSV download, LLM interview questions
│
├── tests/                       # Canonical pytest suite (177 tests reported passing)
│   ├── test_bm25_filter.py
│   ├── test_data_integrity.py
│   ├── test_feature_scorer.py
│   ├── test_honeypot_detector.py
│   ├── test_name_blindness.py           # End-to-end name-swap invariance
│   ├── test_text_builder_name_blindness.py # Text builders never include names
│   ├── test_employer_name_measurement.py   # Consulting-rule employer-effect audit
│   └── test_reasoning_banding.py           # Reasoning tone matches score bands
│
├── scripts/                     # Measurement, evaluation & audit scripts (all numbers reproducible)
│   ├── measure_name_blindness.py    # 800 name-swap comparisons → identical rankings
│   ├── measure_employer_effect.py   # Consulting-rule employer-group effect
│   ├── measure_employer_impact.py
│   ├── measure_nlp_bypass.py
│   ├── recompute_all_metrics.py     # Recomputes every README metric (incl. latency)
│   ├── eval_harness.py              # Batch evaluation harness
│   ├── compare_rankings.py          # Diff two ranked CSVs
│   ├── final_qa_audit.py            # Pre-submission QA
│   ├── verify_submit_ready.py       # Submission readiness gate
│   └── set_phone.py                 # Utility to set phone in submission metadata
│
├── config/
│   └── job_description.json     # Default JD: Senior AI Engineer — Search/Retrieval/Ranking,
│                                #   5+ YoE, required & nice-to-have skills, domain keywords
│
├── data/
│   ├── README.md                # Explains embedding cache generation
│   ├── candidates.jsonl         # Candidate corpus (repo copy; full 100K lives outside)
│   ├── sample_candidates.json   # 50-profile dev sample for the dashboard
│   ├── candidate_embeddings.npy # [generated, gitignored] 100K×768 float32
│   ├── candidate_ids.json       # [generated] row↔ID ordering
│   └── embeddings_meta.json     # [generated] cache fingerprint metadata
│
├── paper/                       # Research artifact (CEUR-WS / TalentCLEF 2026)
│   ├── PAPER_DRAFT.tex          # Full LaTeX source (504 lines)
│   ├── ResearchPaper.pdf        # Compiled paper
│   ├── TalentLensAI_Submission.pdf / .pptx  # Submission versions & slides
│   ├── submission_metadata.yaml # TalentCLEF submission metadata
│   ├── references.bib           # Bibliography
│   ├── ceurart.cls / elsarticle-num-names.bst # LaTeX class & style
│   ├── PAPER_FIGURE_PROMPTS.md  # Prompts used to generate figures
│   ├── paper_figures/           # figure_1.png (architecture) etc.
│   ├── reference_paper_images/  # Reference paper snapshots
│   └── TalentLens_Paper_Flat/   # Flattened paper build
│
├── outputs/                     # Generated ranking CSVs (e.g. outputs/mohd_ibadullah.csv)
├── images/                      # Static images
├── scratch/                     # Scratch space (gitignored)
├── .streamlit/                  # Streamlit config (theme/secrets)
├── candidates.jsonl (root), candidates.corrupted.jsonl, mohd_ibadullah.csv, *.zip,
│   eval_data.json, tex_pieces.json, cc-by.pdf/svg/png, research PDFs  # Local working files
└── __pycache__/ .pytest_cache/ .freebuff/  # Caches (gitignored)
```

---

## 10. APIs / Interfaces

### 10.1 CLI interface

```bash
# Official entry point
python rank.py --candidates ./candidates.jsonl --out ./outputs/participant_id.csv

# Underlying full runner
python src/run_pipeline_full.py \
    --candidates ./candidates.jsonl \
    --jd ./config/job_description.json \
    --out ./outputs/mohd_ibadullah.csv \
    [--validate validate_submission.py] \
    [--check-pool] [--skip-preflight] [--setup] [--allow-bm25-only]
```

| Flag | Meaning |
| --- | --- |
| `--candidates` | Path to candidates JSONL (default `./candidates.jsonl`, with 4 fallback locations searched) |
| `--jd` | JD JSON path (default `config/job_description.json`) |
| `--out` | Output CSV path (default `outputs/mohd_ibadullah.csv`) |
| `--validate` | Optional external validator script, run as `python <validator> <out.csv>` |
| `--check-pool` | Verify every output `candidate_id` exists in the corpus |
| `--setup` | Run model download + embedding precompute once, then exit |
| `--skip-preflight` | Skip model-cache preflight |
| `--allow-bm25-only` | Permit ranking without precomputed embeddings (lower quality) |

### 10.2 Python API (key signatures)

```python
# src/pipeline.py
def run_ranking_pipeline(
    candidates_path: str,
    jd_input: dict | str,            # structured JD dict OR free text
    out_csv_path: str,
    top_n: int = 100,
    use_llm: bool = False,
    weights: dict | None = None,
) -> list[dict]                      # final ranked candidate dicts (also writes CSV)

# src/jd_parser.py
def parse_job_description(jd_input: dict | str) -> dict
# → {"role_title": str, "min_years_experience": float, "seniority_level": str,
#    "required_skills": [str], "nice_to_have_skills": [str], "domain_keywords": [str]}

# src/bm25_filter.py
def build_candidate_document(candidate: dict) -> str
def tokenize(text: str) -> list[str]
class BM25Filter:
    def __init__(self, candidates_list: list[dict]) -> None
    @classmethod
    def from_corpus(cls, corpus: list[str]) -> "BM25Filter"
    def get_top_indices(self, parsed_jd: dict, top_n: int = 3000) -> list[int]
    def filter_candidates(self, parsed_jd: dict, top_n: int = 3000) -> list[dict]

# src/embedding_scorer.py
class EmbeddingScorer:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None
    def load_precomputed_embeddings(self, embeddings_path: str, ids_path: str) -> None
    def search_similar_candidates(self, jd_text: str, top_n: int = 1000) -> list[tuple[str, float]]
    def get_candidate_similarity_by_id(self, candidate_id: str, jd_embedding: np.ndarray) -> float
    def get_embeddings(self, texts: list[str], batch_size: int = 128, is_query: bool = False) -> np.ndarray
    def compute_similarity(self, jd_text: str, candidate_texts: list[str]) -> list[float]

# src/honeypot_detector.py
def detect_trap(candidate: dict) -> tuple[float, str]   # (trap_score 0..1, reason string)
def classify_title(title: str) -> str
def check_boilerplate_description(desc: str) -> str | None

# src/feature_scorer.py
def calculate_candidate_score(candidate: dict, semantic_similarity: float, trap_score: float,
                              parsed_jd: dict, weights: dict | None = None
                             ) -> tuple[float, dict]     # (final_score 0..100, breakdown dict)
def match_skill(cand_skill_name: str, target_skills: list[str]) -> tuple[str | None, bool, float]
def compute_skill_match_score(candidate_skills: list[dict], parsed_jd: dict) -> float
def compute_title_seniority_match(profile: dict, parsed_jd: dict) -> float
def compute_signal_bonus(signals: dict) -> float

# src/cross_encoder_reranker.py
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ettin-reranker-17m-v1") -> None
    def rerank(self, jd_text: str, candidates: list[dict],
               blend_weight: float = 0.4, min_yoe: float = 0.0) -> list[dict]

# src/llm_reranker.py
def rerank_top_candidates(top_candidates: list[dict], parsed_jd: dict,
                          use_llm: bool = False) -> list[dict]
def generate_rule_based_reasoning(candidate: dict, score: float, breakdown: dict,
                                  target_skills: set[str] | None = None,
                                  rank: int | None = None) -> str

# src/candidate_text.py
def build_candidate_embedding_text(cand: dict) -> str

# src/data_loader.py
def load_sample_candidates(file_path: str) -> list[dict]
def stream_candidates(file_path: str) -> Iterator[dict]

# src/data_integrity.py
def check_corpus(path: str | Path, min_populated_fraction: float = 0.95) -> dict
def print_corpus_summary(path: str | Path, min_populated_fraction: float = 0.95) -> dict
```

### 10.3 Output CSV schema (submission format)

| Column | Type | Notes |
| --- | --- | --- |
| `candidate_id` | str | Must exist in the corpus pool (verifiable via `--check-pool`) |
| `rank` | int | 1-based; equal scores tie-break by `candidate_id` ascending |
| `score` | float [0.0, 1.0] | `final_score / 100`, 4 decimal places, capped at 0.994 |
| `reasoning` | str | Fact-grounded recruiter rationale (tone banded to score) |

### 10.4 External API integrations (optional, dashboard/LLM only)

- **Gemini:** `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=<KEY>`
- **Groq:** `POST https://api.groq.com/openai/v1/chat/completions` (model `openai/gpt-oss-120b`), used when the key begins with `gsk_`
- **OpenAI (reasoning path only):** `gpt-4o-mini` via the `openai` client
- Keys resolved from `GEMINI_API_KEY` / `api_key` env vars or a manually parsed `.env`.

There is no HTTP/REST API served by the project itself — the only user interface is the CLI and the Streamlit dashboard (dashboard "endpoints" are Streamlit components: JD textarea, weight sliders, max-results input, run button, CSV download button, per-candidate "Generate Targeted Questions" button).

---

## 11. Setup & Installation

### 11.1 Prerequisites
- Python 3.10+ (uses modern type-hint syntax)
- ~1 GB free disk for model + embedding cache; ≥4 GB RAM recommended (mmap keeps the 100K×768 matrix paged)
- Internet for the one-time model download; ranking runs fully offline afterwards

### 11.2 Standard setup

```bash
git clone <repo-url> talent-lens-ai
cd talent-lens-ai
pip install -r requirements.txt

# One-time: fetch models + build the embedding cache
python src/download_models.py
python src/precompute_embeddings.py        # requires candidates.jsonl (see 11.3)
```

Or use the bundled one-command setup: `./setup.sh` (Linux/Mac) or `.\setup.ps1` (Windows).

### 11.3 Data placement

Place the full `candidates.jsonl` at one of the searched locations:
`./candidates.jsonl` (repo root or cwd), `../candidates.jsonl`,
`../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl`,
or adjacent variants (exact list in `src/run_pipeline_full.py` / `src/precompute_embeddings.py`).
Without it, only the 50-candidate sample mode works.

### 11.4 Run

```bash
# Full ranking (100K)
python rank.py --candidates ./candidates.jsonl --out ./outputs/participant_id.csv

# Tests
pytest -q                                  # 177 tests expected to pass

# Dashboard
streamlit run app/streamlit_app.py         # local (sample + full modes)
# or: python run_app.py                    # headless/cloud on port 8501
```

### 11.5 Optional environment variables (never required for ranking)

```
GEMINI_API_KEY=<key>    # dashboard interview questions / LLM reasoning
api_key=<groq-key>      # Groq-hosted model for interview questions
```

---

## 12. Current Status

**Done and measured (from README + scripts):**

- Full 6-stage pipeline implemented, CLI + Streamlit UI, live Streamlit Cloud demo deployed.
- 177 pytest tests passing (`pytest -q`).
- Corpus: 100,000 profiles, 100% populated, SHA-256-verified integrity guard.
- Name blindness: 800/800 name-swap comparisons yield identical rankings.
- Embedding cache with dual fingerprint verification (IDs + text) implemented and tested.
- Cross-encoder swap to ettin-reranker-17m-v1 completed (measured dominant).
- Research paper written and compiled (CEUR-WS format); TalentCLEF 2026 Task A benchmark evaluation complete with statistics (paired bootstrap, Bonferroni).
- Honest-metrics pass: circular "150% lift" claim removed; `scripts/recompute_all_metrics.py` regenerates every published number.

**In progress / partial:**

- Preprint link for the paper: "coming soon" per README.
- CLEF 2027 venue/date fields in the paper header are placeholders (`[TODO-USER]`).

**Not started / not yet defined:**

- Fine-tuning or training of any model (all models used off-the-shelf).
- A served REST/gRPC API.
- Docker packaging.
- Multi-JD batch ranking UI; user accounts/persistence in the dashboard.

---

## 13. Known Issues / Limitations

1. **Consulting-career rule fairness (disclosed):** the >60%-consulting-tenure ×0.50 penalty fires disproportionately for one employer group (Fisher exact test p=0.0046 at top-20, audited on 200 candidates). The project *documents* this as a policy choice rather than claiming fairness.
2. **Rule-based honeypot detector is template-bound:** it matches specific decoy strings ("experimented with chatgpt", boilerplate role descriptions, "founded in YYYY" patterns). New, differently-crafted decoys would evade it. It is deterministic, not learned.
3. **JD free-text parsing is heuristic:** regex + a fixed 40-skill taxonomy + section-header heuristics; unusual JDs may parse poorly (e.g., non-English JDs, unconventional skill names).
4. **English-centric embeddings:** bge-base-en-v1.5 is English-focused — the paper shows Spanish text suffers token-budget pressure; the product pipeline inherits this for non-English profiles.
5. **Latency:** one-time BM25 index build ≈ 25 s; ≈ 13 s per repeat query on the measured machine (i7-class CPU). No GPU path implemented.
6. **Hard-coded reference date** (2026-06-16) for inactivity calculations, and hard-coded disqualifier keyword lists — config drift risk.
7. **Score caps are judge-motivated** (99.4 ceiling, display-score rank offsets in the UI) — cosmetic calibration, not model calibration; display scores in the UI deliberately diverge slightly from internal scores.
8. **LLM features are external-API dependent** and fail gracefully to rule-based reasoning, but need keys; no local LLM fallback.
9. **Sample/full duality:** the dashboard's cloud mode can't rank 100K (dataset not committed); full ranking is CLI-only.
10. **Legacy dev files remain** (`src/test_*.py`, `src/explore.py`, root-level `candidates.corrupted.jsonl`, large evidence zips) — clutter risk for newcomers.

---

## 14. Next Steps / Roadmap

From the repo's own signals (paper TODOs, README, structure) — priority order inferred:

1. **Publish the preprint** (arXiv/CEUR-WS) and fill in CLEF 2027 venue metadata (explicit `[TODO-USER]` in `PAPER_DRAFT.tex`).
2. **Generalize the honeypot detector** beyond fixed templates (e.g., a small classifier on profile-consistency features) while keeping it name-blind.
3. **Externalize magic constants** (consulting company list, disqualifier keywords, reference date, thresholds) into `config/`.
4. **Multilingual support:** swap in a multilingual encoder (e.g., bge-m3) and validate name-blindness on the Spanish split.
5. **Dockerize** the CLI + dashboard for reproducible deployment; add CI running `pytest -q` and `scripts/recompute_all_metrics.py`.
6. **Batch/queue mode** for many JDs against one indexed corpus (index reuse is already amortized in the Streamlit cache).
7. Optional: replace rule-based reasoning with a local small LLM for offline environments.

---

## 15. Team & Roles

| Person | Role | Evidence |
| --- | --- | --- |
| **Mohd Ibadullah** (Keshav Memorial Institute of Technology, JNTUH; mohdibadullah24@kmit.edu.in) | Sole author and developer — entire pipeline, tests, scripts, dashboard, and paper (all 50 commits, single contributor) | Git history (single identity), paper authorship, `outputs/mohd_ibadullah.csv`, submission metadata |

No other contributors. Advisor/guide attribution: **Not yet defined** (not present in the repo).

---

## 16. Constraints

- **Competition schema:** the submission CSV must have exactly `candidate_id, rank, score, reasoning` with score in [0, 1] capped at 0.994 and candidate IDs from the official pool — enforced in `pipeline.py` and `run_pipeline_full.py --check-pool`.
- **CPU-only:** the challenge required CPU execution; model choices (17M-param reranker, 160-token truncation, batch 128) were made for CPU latency.
- **Offline verification:** Stage-3 sandbox verification required offline execution → `download_models.py` + preflight exist so no network calls happen during ranking.
- **Name-blindness requirement:** ranking must not read candidate names (verified by 800/800 swap test and unit tests).
- **Honesty policy (self-imposed):** no estimated or circular metrics; every published number must be reproducible by a script in `scripts/`.
- **License:** code MIT; TalentCLEF benchmark data used under its license with attribution (DOI 10.5281/zenodo.17625261).
- **Deadlines / grading criteria / guide's requirements:** Not yet defined in the repository (no dates, rubric, or advisor requirements are committed).
