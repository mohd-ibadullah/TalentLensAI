# TalentLens AI — Interview Preparation Guide & Pipeline Story

This guide prepares you to pitch and defend the **TalentLens AI** candidate ranking engine in front of hackathon judges and technical recruiters.

---

## 🎯 The Core Pitch: Recruiter Intent over Keyword Stuffing
> **"We didn't build a keyword matcher. We built a recruiter-aligned pipeline that prioritizes career depth over profile stuffing."**

*   **The Problem:** Traditional applicant tracking systems (ATS) rely heavily on keyword frequencies. Candidates exploit this by listing 50+ machine learning skills in their summaries, causing keyword-stuffing tools to rank them highly even if they lack core experience.
*   **Our Solution:** We systematically minimized keyword bias. We capped direct skill-overlap matching at **20%** and amplified semantic context (**40%**) and career seniority (**20%**).
*   **The Outcome:** The ranking is governed by actual career history, not resume tailoring. Keyword-stuffed fake profiles are automatically penalized, while experienced candidates with rich career histories rise to the top.

---

## 🏗️ 6-Stage Hybrid Architecture

| Stage | Name | Input ➔ Output | Tech Stack & Rationale |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **BM25 Lexical Filter** | 100K ➔ 1K profiles (~8s) | `rank-bm25` (Okapi). Extremely fast, streams profiles in a single pass to stay within memory limits. |
| **Stage 2** | **Honeypot Trap Detector** | Flags & blocks decoys | Checks timeline anomalies (starting a role before company founded) and template-matching. |
| **Stage 3** | **Semantic Embedding Scorer** | Dense cosine similarity | `BAAI/bge-base-en-v1.5` (local on CPU). Computes deep similarity on enriched profile text. |
| **Stage 4** | **Feature Scoring Engine** | Combines signals & gates | Weighted scoring + YoE hard gates + career bonuses + behavioral adjustments. |
| **Stage 5** | **Cross-Encoder Reranker** | Top 150 pairwise rerank | `ms-marco-MiniLM-L6-v2`. Computes deep query-document relevance to capture hidden context. |
| **Stage 6** | **Reasoning Generator** | Rank-aware CSV reasoning | Programmatic, non-templated text detailing actual skills matched and experience flags. |

---

## 📊 The Scoring Formula & Weights (40 / 20 / 20 / 10)

Your final candidate ranking score is computed via a rigorous ensembled scoring engine:

$$\text{Raw Positive Score} = 100 \times \left( \frac{0.40 \times \text{Semantic} + 0.20 \times \text{Skills} + 0.20 \times \text{Title/YoE} + 0.10 \times \text{Signals}}{0.90} \right)$$

$$\text{Final Score} = \text{Raw Positive Score} - 40 \times \text{TrapScore} - \text{YoE Penalty} + \text{Career Bonus} - \text{Disqualifiers} + \text{Behavioral Adj.}$$

### Defending the Weights:
*   **40% Semantic Similarity:** Connects conceptual intent (e.g., mapping "semantic search" to "vector index implementation").
*   **20% Experience & Title Relevance:** Direct check on seniority (protects the 5.0+ YoE mandate).
*   **20% Skills Overlap:** Reduced from 30% to suppress keyword-stuffer bias, normalized with RapidFuzz token matching.
*   **10% Behavioral Signals:** Amplifies active job seekers with fast response rates without penalizing missing data (-1 values).

---

## 🚫 Gating, Disqualifiers, and Behavioral Envelopes
Be ready to walk the judges through the exact constraints and disqualifiers built into our code:

### 1. Programmatic Experience Gates
*   **YoE < 4.0:** Programmatic **-50.0 points** penalty (100% excluded from the top 100).
*   **4.0 <= YoE < 5.0:** Programmatic **-15.0 points** penalty (excluded from the top 50, allowed in top 100).
*   **YoE >= 5.0:** Unconstrained. Resulting top 8 candidates have **5.2 to 8.9 YoE**.

### 2. Industry & Profile Disqualifiers (-20 Points Flat Penalty)
*   **Consulting-Heavy Penalty (>60% duration):** Candidates spending >60% of their career at IT consultancies (TCS, Wipro, Infosys, Cognizant, etc.) are penalized to prioritize product development experience.
*   **Pure CV/Speech/Robotics:** If a candidate has CV/Speech/Robotics keywords but **zero** NLP, Search, or Information Retrieval terms, they are penalized.
*   **LangChain-only Wrappers:** Candidates mentioning LangChain but lacking core ML libraries (PyTorch, TensorFlow, scikit-learn) and carrying **zero** historical ML/AI job titles are penalized as "wrapper-only" developers.
*   **Title/Skills Stuffing Check:** HR, sales, or marketing titles that copy-paste AI keywords into their summary get immediately flagged and penalized.

### 3. Behavioral Adjustments
*   **Inactivity Penalty (-10 points):** Profile idle for >180 days relative to active search date.
*   **Low Response Rate (-10 points):** Recruiter response rate below 15%.
*   **Notice Period Penalty (-5 points):** Candidates with notice period > 60 days.
*   **Open-to-work Bonus (+5 points):** Candidates marked open-to-work with response rate >= 70%.

---

## 🛡️ Adversarial Honeypot Resilience
*   **The Baseline Vulnerability:** The BM25 baseline index contains **30.1% honeypot decoys** in its top 1,000 matches.
*   **Our Solution:** The Stage 2 Honeypot Detector analyzes timeline anomalies (e.g. role start date predating company establishment) and flags copy-paste description templates.
*   **The Verification:** Out of the final 100 output candidates, **0.0%** honeypot profiles slipped through.

---

## 🚀 Key Takeaways for the Q&A
1.  **"Why BGE-base instead of a larger model?"**
    *   *Answer:* Efficiency and speed. BGE-base-en-v1.5 has an index footprint that fits comfortably in CPU memory, yielding ~144 seconds total run time, whereas larger models would exceed the 5-minute timeout constraint on 100K profiles.
2.  **"How do you handle missing values (-1) in Redrob signals?"**
    *   *Answer:* Fairness-first. If a signal is missing (value = -1), the system ignores it and calculates the average from remaining signals, ensuring we do not penalize candidates for missing platform telemetry.
3.  **"Is the system truly fair?"**
    *   *Answer:* Yes. The parsing and scoring phases explicitly strip out candidate names, gender indicator words, location, and college names, ensuring pure merit-based matching.
