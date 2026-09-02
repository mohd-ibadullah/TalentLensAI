# TalentLens AI

**AI-powered candidate discovery and ranking for 100,000 résumés, on CPU only, with**
**verified name-blind ranking and a published research study behind it.**

Python 3.10+ · 176 tests passing · CPU only · CC-BY-4.0

---

## What it does

TalentLens ranks a candidate pool against a job description in three stages:

1. **Retrieval** — BM25 (lexical) fused with dense semantic search (`bge-base-en-v1.5`).
2. **Screening** — a honeypot detector that flags keyword-stuffed and decoy profiles.
3. **Scoring + reranking** — weighted skill match, title relevance, recruiter signals, and a
   cross-encoder rerank (`ettin-reranker-17m-v1`).

It runs end to end on a standard CPU, with a Streamlit dashboard for interactive use.

![Architecture](paper_figures/figure_1.png)

---

## Measured results (no fabricated numbers)

Everything below is measured and reproducible. The scripts that produce these numbers are
in `scripts/`.

| Property | Measured value | How it was verified |
|---|---|---|
| Corpus | 100,000 profiles, 100% populated | SHA256 integrity guard |
| Name-blindness | 800 / 800 name swaps produce identical rankings | `scripts/measure_name_blindness.py` |
| Honeypot detection | decoys removed from the shortlist | `tests/` + honeypot suite |
| Test suite | 176 tests passing | `pytest -q` |
| Latency (components) | BM25 index ~25 s one-time; ~13 s per repeat query | `scripts/recompute_all_metrics.py --latency` |
| Cache safety | embeddings carry a text fingerprint, so stale vectors are detected and refused | `src/pipeline.py` |

**Why name-blindness matters:** the pipeline never reads a candidate's name when ranking.
Changing the name while keeping every other byte of the profile identical leaves the score
and rank unchanged in all 800 measured comparisons. The one place employer names are read
is a documented consulting-career rule, disclosed below.

---

## Research

This repository is backed by a measured study on **identity leakage in résumé retrieval**:

> **Where Does the Name Leak? A Stage-Wise Decomposition of Identity Sensitivity in
> Résumé Retrieval** — Mohd Ibadullah.
>
> On the TalentCLEF 2026 benchmark, dense retrieval reorders candidates when only the name
> changes (RBO 0.669 EN / 0.639 ES), while BM25 stays almost name-blind. Deleting the
> identity header closes the gap to RBO 1.000 at no measurable accuracy cost.

- Benchmark: TalentCLEF 2026 Task A (CC-BY-4.0, DOI 10.5281/zenodo.17625261)
- Full study + reproducibility package: `talentlens-study/`
- Paper: `paper_draft.tex` ([PREPRINT LINK — TODO once uploaded])

---

## Fairness and honesty

- **Candidate names are excluded from ranking.** Display names exist only in the UI.
- **The consulting-career rule is documented, not hidden.** Candidates whose career
  history is more than 60% at consulting firms are penalised. A 200-candidate audit shows
  this rule fires disproportionately for one employer group (Fisher exact, p = 0.0046 at
  top-20). This is disclosed as a policy choice, not presented as a fairness guarantee.
- **No pseudoscience metrics.** Earlier internal drafts claimed "+150% lift" from a
  circular evaluation. Those numbers were wrong and have been removed. What remains is
  what can be measured honestly.

---

## Tech stack

`Python` · `rank_bm25` · `sentence-transformers` (`bge-base-en-v1.5`) ·
`cross-encoder` (`ettin-reranker-17m-v1`) · `RapidFuzz` · `Streamlit` · `pytest`

---

## Quick start

```bash
cd talent-lens-ai
pip install -r requirements.txt
python src/download_models.py          # one-time, needs network
python src/precompute_embeddings.py    # one-time, ~15 min CPU

# rank the full corpus
python rank.py --candidates ./candidates.jsonl --out ./outputs/<participant_id>.csv

# run the tests
pytest -q

# interactive dashboard
streamlit run app/streamlit_app.py
```

---

## Repository layout

```
src/        pipeline, retrieval, scoring, reranking, data integrity guard
app/        Streamlit dashboard
tests/      unit + integration + fairness tests (176 passing)
scripts/    evaluation, measurement, and audit scripts
talentlens-study/   the TalentCLEF identity-leakage study (reproducible)
paper_figures/      publication figures used in the paper
```

---

## Live demo

https://talentlensai-nxrk7zxjmaxvnwnubyvz7n.streamlit.app/

---

## License

Code and study are released under CC-BY-4.0. Benchmark data remains under the TalentCLEF
license (CC-BY-4.0) with attribution.
