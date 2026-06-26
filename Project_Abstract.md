# Project Abstract

**Project Title:** A Hybrid Multi-Stage Retrieval Pipeline for Fair and Adversarial-Resilient AI-Based Candidate Ranking

**Base Paper:** Dharmendra, T. A., Meenakshi, K., & Bhargava, D. (2025). "NLP-Powered Resume Screening and Ranking System." 2025 3rd International Conference on Disruptive Technologies (ICDT), pp. 1361-1366, IEEE. DOI: 10.1109/icdt63985.2025.10986338

**Team Members:**
- Mohd Ibadullah (Team Lead)

**Guide Name:** ____________________________

**Guide Signature:** ____________________________

---

## Abstract

The rapid growth of online recruitment platforms has led to an overwhelming volume of candidate profiles, making manual screening impractical and error-prone. Traditional Applicant Tracking Systems (ATS) rely on keyword matching, which fails to capture contextual relevance and is highly vulnerable to adversarial keyword-stuffing attacks by deceptive applicants. This project proposes TalentLens AI, a hybrid multi-stage candidate ranking pipeline that addresses these limitations through a 6-stage architecture designed to process 100,000 candidate profiles on CPU-only hardware within 140 seconds.

The proposed system operates as follows: Stage 1 employs BM25 lexical indexing for fast coarse retrieval, narrowing the candidate pool from 100,000 to 1,000 profiles. Stage 2 introduces a novel rule-based Honeypot Trap Detector that identifies keyword-stuffed decoy profiles by cross-checking career history consistency, boilerplate summary templates, and skill-title mismatches. Stage 3 computes dense semantic similarity using the BAAI/bge-base-en-v1.5 bi-encoder transformer model (768-dimensional embeddings). Stage 4 applies a Cross-Encoder reranker (ms-marco-MiniLM-L6-v2) for fine-grained contextual relevance scoring on the top 150 candidates. Stage 5 combines all signals through a weighted ensemble formula incorporating semantic similarity (40%), skill matching via RapidFuzz (20%), title and experience alignment (20%), and Redrob behavioral engagement signals (10%), with a trap penalty of up to 40% for flagged profiles. Stage 6 generates rank-aware candidate justifications and produces the final validated output.

The system achieves a +150.0% relative lift in Precision@10, a +133.2% relative lift in NDCG@10, and a 0.0% honeypot infiltration rate in the top 100 candidates over a BM25-only baseline. The framework ensures fairness by explicitly excluding candidate names, gender, college tier, and geographic location from the scoring pipeline, following established algorithmic hiring best practices. An interactive Streamlit dashboard enables real-time weight tuning and candidate exploration.

**Keywords:** Information Retrieval, Resume Ranking, BM25, Semantic Embeddings, Cross-Encoder Reranking, Adversarial Robustness, Fairness in AI Hiring, NLP

---

**Future Scope:**
- SHAP/LIME-based explainability for transparent scoring decisions
- Domain-specific fine-tuning of embedding models on HR/recruitment data
- ML-based resume fraud detection using supervised anomaly models
- Multi-language JD and resume parsing for regional hiring markets
