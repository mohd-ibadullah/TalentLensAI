# Where Does the Name Leak? A Stage-Wise Decomposition of Identity Sensitivity in Résumé Retrieval

Mohd Ibadullah
Keshav Memorial Institute of Technology, JNTUH
REDACTED

**Draft v0.2 — working notes, target: CLEF 2027 (CEUR-WS) · 6–8 pages**
**Status: all numbers below are measured; the two former `[PENDING-ZIP]` items (H1 Δ CIs, ES AP significance) are now resolved.**

---

## Abstract

Résumé-ranking systems increasingly rely on dense retrieval, yet the extent to which a
candidate's name influences their rank is rarely measured. We decompose identity leakage
stage by stage in a hybrid retrieval pipeline evaluated on the TalentCLEF 2026 Task A
benchmark (English and Spanish, 472 CVs, official human relevance judgments). Holding
every other byte of a CV constant and substituting only the candidate's name across four
name-origin groups and two genders, we find a monotonic leakage gradient: BM25 is almost
name-blind (Rank-Biased Overlap 0.99), while dense retrieval with `bge-base-en-v1.5`
reorders substantially (RBO 0.669 EN / 0.639 ES); hybrid fusion and cross-encoder
reranking fall in between (0.773 and 0.899 EN). The BM25–dense gap is on the order of
0.3–0.4 RBO, significant under a paired bootstrap with Bonferroni correction. Removing the
identity header — three lines of preprocessing, no model change, no additional compute —
raises RBO to exactly 1.000 in both languages. The accuracy cost is statistically
unresolvable in English (all p > 0.008 after correction); in Spanish the same intervention
measurably *improves* dense retrieval (ΔAP +0.094, 95% CI [+0.041, +0.146], p = 0.0003
after correction) — a gain we attribute to token-budget relief for an English encoder
over Spanish text, not to the name itself. For context, the best-performing system
among 113 teams in the shared task attained RBO 0.9904 on the held-out test split using a
large-parameter ensemble. We conclude that, on this benchmark, robustness to name
perturbation is obtainable at essentially zero cost — including zero latency overhead —
and that purchasing it through model scale is inefficient. Code, data DOIs and every
per-query measurement are released.

---

## 1. Introduction

The TalentCLEF 2026 overview reports that none of the top systems explicitly include a
dedicated bias-mitigation component, and the organizers state that the bias-track results
"should be interpreted as evidence of ranking robustness under the specific
masculine/feminine variants used in the benchmark, rather than as a comprehensive
demonstration of fairness or bias mitigation" [Gasco et al., 2026]. They further
hypothesise that the winning system's skill/task extraction reduced surface-lexical
influence — but that hypothesis was never tested.

**We test it.** We ask a narrower, falsifiable question: *given a fixed CV, how much does
the candidate's name alone move their rank, and where in the pipeline does that movement
happen?*

Contributions:

1. The first stage-wise identity-leakage decomposition for résumé retrieval (BM25 → dense
   → hybrid → cross-encoder rerank).
2. An extension of the benchmark's gender-only probe to **name origin × gender**.
3. A zero-cost intervention (identity-header stripping) with its accuracy cost quantified
   and confidence-bounded.
4. A reproducible artifact: CC-BY data, DOI + SHA256, fixed seeds, one command.

---

## 2. Related Work

- **Embedding-based résumé screening bias** — Wilson & Caliskan (AIES 2024) show masked
  language models favour White-associated names and disadvantage Black male names, measured
  as *selection rate on a private corpus*. We measure *rank stability on a public,
  qrel-backed benchmark*, which makes the counterfactual exact and reproducible.
- **Cross-cultural LLM hiring bias** — prior work on Indian vs UK transcripts (arXiv
  2508.16673) documents surface-level name effects in generative hiring models.
- **Human-AI propagation** — biased recommendations have been shown to shift human choices
  (arXiv 2509.04404); "a human reviews it" is not by itself a defence.
- **Benchmark & metric** — TalentCLEF 2025/2026 overviews; Rank-Biased Overlap
  (Webber, Moffat & Zobel, TOIS 2010), the metric the shared task itself uses for bias.
- **Weak baselines** — Yang, Lu & Lin (SIGIR 2019) and Armstrong et al. (2009) motivate the
  BM25-first framing: strong lexical baselines are the correct reference point.
- **Efficiency-aware IR** — the broader argument that robustness/quality should be bought
  with measured compute, not assumed to require scale.

---

## 3. Experimental Setup

**Benchmark.** TalentCLEF 2026 Task A dev split, English and Spanish: 10 queries, 472 CVs,
472 qrels each. CC-BY-4.0. Concept DOI `10.5281/zenodo.17625261`, version
`10.5281/zenodo.19652670`. Identities are synthetic (not redactions of real people).

**A corpus property that makes the counterfactual exact** (the methodological backbone of
the paper, and we report it as a measured fact):

- 472/472 CVs carry the name on line 1
- **0/472** contain any gendered pronoun (he/she/his/her/him)
- **0/472** contain gendered job titles

Consequently the name is the *only* identity signal in the text: swapping one line changes
nothing else. This is what lets us attribute rank movement to the name alone.

**Pipeline (four stages).**

| stage | implementation |
|---|---|
| BM25 | `rank_bm25` over the concatenated profile |
| Dense | `BAAI/bge-base-en-v1.5`, max 512 tokens |
| Hybrid | min-max normalised sum, α = 0.5 (not RRF — the score magnitudes are needed downstream) |
| Rerank | `cross-encoder/ettin-reranker-17m-v1`, top-150 |

**Perturbations.** 4 name-origin groups (Anglo, Indian, African, Arabic) × 2 genders, plus
a round-robin `mixed` condition for impact-ratio computation. 9 conditions total.

**Interventions.** `none` · `strip-identity` (delete the header block) · `mask-name`
(replace the name token with a placeholder) · `skill-only` (retain only skills).

**Metrics.** MAP, nDCG@10, R@50 (via `ir_measures`) · RBO with p = 0.9 (the organizers'
metric) · selection impact ratio against the EEOC four-fifths threshold (descriptive only).

**Statistics.** Paired bootstrap B = 10,000, seed 42, Bonferroni correction within each
family. Per-query results are released. **n differs by metric, and we state it explicitly:**
AP significance is paired per query (n = 10 per language); RBO significance is paired over
the 9 perturbation conditions per query (n = 90 per stage comparison). With n = 10 queries
per language, the declared AP resolution floor is ±0.07–0.12 — effects below this are
*not resolvable*, and we treat them as such rather than as zero. The Bonferroni-corrected
α for AP is 0.0021 (0.05/24, family = all intervention-vs-none AP comparisons).

---

## 4. Results

### 4.1 H1 — leakage is monotonic in lexical content

Rank stability under name-only perturbation (RBO, 1.0 = no effect). Post-fix values from
the full GPU run:

| stage | RBO EN | RBO ES |
|---|---|---|
| BM25 | ~0.99 | ~0.998 |
| Rerank | 0.899 | 0.860 |
| Hybrid | 0.773 | 0.772 |
| Dense | **0.669** | **0.639** |

The ordering — BM25 most stable, dense least stable, hybrid and rerank in between — is
monotonic in how much the stage relies on dense vector similarity versus exact lexical
match, and it replicates across both languages. Paired-bootstrap Δ vs BM25 (post-fix,
30/30 significant, Bonferroni α = 0.001667, n = 90 per comparison):

| comparison | ΔRBO | 95% CI |
|---|---|---|
| EN: BM25 → Dense | +0.3269 | [0.3045, 0.3504] |
| EN: BM25 → Hybrid | +0.2201 | [0.2022, 0.2381] |
| EN: BM25 → Rerank | +0.0925 | [0.0818, 0.1037] |
| ES: BM25 → Dense | +0.3615 | [0.3359, 0.3875] |
| ES: BM25 → Hybrid | +0.2269 | [0.2074, 0.2472] |
| ES: BM25 → Rerank | +0.1386 | [0.1268, 0.1502] |

### 4.2 H2 — the intervention closes the gap completely

| stage | ΔRBO EN (strip-identity) | ΔRBO ES (strip-identity) |
|---|---|---|
| Dense | +0.3351 [0.3132, 0.3585] | +0.3629 [0.3374, 0.3889] |
| Hybrid | +0.2283 [0.2104, 0.2462] | +0.2283 [0.2090, 0.2484] |
| Rerank | +0.1007 [0.0901, 0.1116] | +0.1400 [0.1282, 0.1516] |

9/9 comparisons significant in each language (paired bootstrap, Bonferroni).
`strip-identity` and `skill-only` reach **exactly 1.0000** in both languages; `mask-name`
reaches 0.9976–0.9999, capped by a documented 1.06% residual (5/472 documents where the
name occurs below the header and survives masking).

### 4.3 The accuracy cost is not resolvable — and in one setting it is a gain

**English** — intervention vs `none`, Average Precision:

| stage | intervention | ΔMAP | 95% CI | p | corrected α | verdict |
|---|---|---|---|---|---|---|
| dense | strip-identity | +0.0116 | [−0.0014, +0.0246] | 0.077 | 0.0021 | not sig |
| dense | mask-name | +0.0109 | [+0.0000, +0.0260] | 0.092 | 0.0021 | not sig |
| hybrid | strip-identity | −0.0013 | [−0.0107, +0.0083] | 0.764 | 0.0021 | not sig |
| hybrid | mask-name | +0.0009 | [−0.0032, +0.0058] | 0.717 | 0.0021 | not sig |
| rerank | strip-identity | −0.0240 | [−0.0826, +0.0200] | 0.362 | 0.0021 | not sig |
| rerank | mask-name | −0.0086 | [−0.0243, +0.0053] | 0.257 | 0.0021 | not sig |

**Spanish** — intervention vs `none`, Average Precision (Bonferroni α = 0.0021, family = 24):

| stage | intervention | ΔMAP | 95% CI | p | verdict |
|---|---|---|---|---|---|
| dense | strip-identity | **+0.0939** | [+0.0411, +0.1460] | **0.0003** | **significant** |
| dense | skill-only | +0.2270 | [+0.1628, +0.2904] | <0.0001 | significant |
| dense | mask-name | −0.0089 | [−0.0419, +0.0262] | 0.626 | not sig |
| hybrid | strip-identity | +0.0145 | [+0.0010, +0.0277] | 0.031 | not sig |
| hybrid | mask-name | −0.0021 | [−0.0160, +0.0107] | 0.762 | not sig |
| rerank | strip-identity | +0.0189 | [−0.0027, +0.0402] | 0.083 | not sig |
| rerank | mask-name | +0.0093 | [−0.0027, +0.0202] | 0.120 | not sig |

The correct wording is **accuracy-neutral or better**. In English the intervention neither
helps nor hurts at resolvable effect sizes (all p > 0.07, α = 0.0021). We state two
*different kinds* of English null explicitly:

- **hybrid** has a tight CI (±0.01) → positive evidence of neutrality;
- **rerank** has a wide CI ([−0.083, +0.020]) → *absence of evidence*; an 8-point drop
cannot be excluded at n = 10. We do not launder this into "neutral".

**The Spanish result — stated with its mechanism caveat.** `strip-identity` significantly
*improves* Spanish dense AP (+0.094, p = 0.0003). This is real in direction, but two
readings must be kept apart:

1. **What we measured:** on a Spanish corpus indexed by an *English* encoder (baseline AP
   0.5487, far below EN's 0.8590), removing the identity header measurably improves
   relevance.
2. **What we do NOT claim:** that the candidate's name *degrades* Spanish retrieval. The
   more plausible mechanism is **token-budget relief** — the header occupies part of the
   512-token window, and for an encoder outside its language every token of actual Spanish
   content matters. Consistent with this, the more aggressive `skill-only` (which removes
   *everything but* skills, freeing maximal budget) helps even more (+0.227). We report the
   effect as measured, attribute it to token budget as a hypothesis, and treat the n = 10
   magnitude with caution.

We do **not** claim "accuracy improves" in English (CIs contain zero, and dense/mask-name
p = 0.092 does not survive correction). The English null — "a free fairness fix" — is the
paper's strongest asset; the Spanish gain is a bonus observation, not the load-bearing
claim.

### 4.4 Cross-lingual asymmetry

Spanish leaks *more* (dense RBO 0.639 vs 0.669 EN) even though its BM25 is *more stable*
(~0.998 vs ~0.99). Spanish dense MAP is also far weaker (0.5487 vs 0.8590 EN), consistent
with applying an **English** encoder to Spanish text. Under `strip-identity`, Spanish dense
MAP rises to 0.6426 (+0.094, p = 0.0003, significant), and under `skill-only` to 0.7756
(+0.227, p < 0.0001, significant). Both directions replicate the RBO finding — the
intervention is never harmful and sometimes corrective — but the *mechanism* (token-budget
relief vs. a genuine name effect) remains an open, testable question; we do not resolve it
here. A natural follow-up is a Spanish-native encoder, which would isolate the two.

### 4.5 Which intervention to deploy

All three interventions reach RBO ≈ 1.0, so RBO does not discriminate among them. The
deciding axis is accuracy cost: `skill-only` costs rerank −0.059 MAP in English (its heavy
content removal has a price at the rerank stage); `strip-identity` costs nothing resolvable
in English and measurably helps Spanish dense retrieval (+0.094). **Recommend
`strip-identity`** — the only intervention that is free everywhere and corrective in at
least one setting.

### 4.6 Latency — the intervention is free by construction, and the CPU bottleneck is elsewhere

CPU-only timing on the English dev set (472 CVs, 10 queries, 512-token window, n = 5
repeats per stage, cache cleared where effective):

| stage | p50 latency/query |
|---|---|
| BM25 | 0.52 s |
| Dense | *not reported* (see below) |
| Hybrid | not informative (dense cache-dependent) |
| Cross-encoder rerank (top-150) | 164.4 s |

Two honest readings:

1. **The intervention adds zero latency.** `strip-identity` is text preprocessing before
   any model runs; its cost is zero by construction, so the robustness gain in §4.2 has no
   runtime price.
2. **The rerank stage dominates CPU cost** — ~300× BM25 — and, per §4.3, that cost buys no
   resolvable accuracy change at n = 10. This is the efficiency framing the paper's
   conclusion relies on: robustness was obtained for free, while the most expensive stage
   contributed nothing measurable.

The dense cold-start latency is **not reported**: the measured p50 (0.014 s) is a cache
hit, i.e. the cache clear was ineffective despite the manifest flag; the p95 (482 s)
reflects a single uncached run with n = 5 and is not a reliable cold-start estimate. We
state this as an open measurement, not a number.

---

## 5. Threats to Validity

1. **n = 10 queries per language.** Effects below the reported half-CI are unresolvable;
   the wide rerank AP CI is explicitly *absence of evidence*.
2. **Dev/test difficulty mismatch.** Untuned BM25 scores MAP 0.8501 on dev; the official
   baseline scores 0.4001 on test. Dev is materially easier; no cross-split comparison is made.
3. **Perturbation completeness.** 9/472 CVs (1.9%) mention the name below the header and
   retain it; measured leakage is therefore a **conservative lower bound**.
4. **mask-name residual.** 5/472 (1.06%) documents remain condition-dependent, capping its
   achievable RBO marginally below 1.0.
5. **Synthetic identities.** Names are surface markers; we make no claim that a name
   determines a person's gender or origin.
6. **RBO measures stability, not fairness.** Same caveat the organizers make about their own
   bias-track evaluation.
7. **One encoder.** Findings are for `bge-base-en-v1.5`; generalisation to other encoders
   is untested.
8. **The Spanish AP gain is not attributed to the name.** We report that `strip-identity`
   improves Spanish dense AP (+0.094, p = 0.0003), but we do *not* claim the name degrades
   Spanish retrieval. The gain is consistent with token-budget relief, is measured at
   n = 10 on a weak baseline (0.5487), and its magnitude carries wide uncertainty
   (CI [+0.041, +0.146]). A Spanish-native encoder is the clean follow-up.
9. **Dense cold-start latency not measured.** Cache clearing was ineffective; the reported
   p50 (0.014 s) is a cache hit and must not be quoted. BM25 and rerank latencies are
   clean (§4.6).

---

## 6. Reproducibility

DOI + SHA256 of both corpora, pinned dependencies, seed 42, one command
(`run_study.py`), per-query CSVs, and a **preflight invariance check** that verifies each
perturbation changes exactly the intended bytes. Zenodo code archive to be minted.

### 6.1 Implementation pitfalls (worth one short paragraph)

Two bugs were caught by the invariance check; both would have produced publishable-looking
nonsense:

1. **Unbounded substitution** deleted the surname `Nair` out of *Nairobi*, `Brad` out of
   *Allen-Bradley*, `Greg` out of *aggregate*.
2. **Non-canonical key ordering** caused a content-addressed embedding cache to return
   vectors misaligned with document ids, producing RBO 0.0329 where the correct answer is
   1.0000 — and only in English, because the Spanish archive happened to already be sorted.

We ship the preflight check as part of the artifact; a two-sentence note of this kind is
the sort of detail that reviewers trust.

---

## 7. Conclusion

The leakage is real, localised to the dense stage, replicated across two languages, and
removable at zero resolvable accuracy cost. A large-parameter ensemble reached RBO 0.9904
on the shared task's test split; a header regex reaches 1.0000. **The field is buying
robustness the expensive way.**

---

## Appendix A — Exact line between honest and not

**Say:**
- "Dense retrieval reorders candidates under name-only perturbation; BM25 does not."
- "The effect replicates across English and Spanish."
- "Removing the identity header eliminates the effect at no resolvable accuracy cost in
  English, and measurably improves Spanish dense retrieval on this benchmark (+0.094 AP,
  p = 0.0003), which we attribute to token-budget relief rather than to the name itself."
- "On this benchmark, robustness does not require scale."

**Do not say:**
- "We show discrimination." (rank instability ≠ discriminatory outcome)
- "Accuracy improves in general." (true only for ES dense, and mechanism-ambiguous)
- "Names degrade Spanish retrieval." (token-budget hypothesis, unproven)
- "Dense retrievers are biased." (one encoder, one benchmark, two languages)
- "Our system beats the leaderboard." (different split, never comparable)
- "This is fair." (only: *more robust under this specific perturbation*)

---

## Appendix B — Former `[PENDING-ZIP]` items — RESOLVED

1. **H1 Δ vs BM25, post-fix** — done. Post-fix paired-bootstrap CIs, 30/30 significant,
   Bonferroni α = 0.001667; values now in §4.1. (Pre-fix estimates were off by ~−0.02 on
   dense; they are not printed anywhere in this draft.)
2. **ES AP significance** — done. ES dense `strip-identity` +0.094 [0.041, 0.146],
   p = 0.0003; `skill-only` +0.227, p < 0.0001; hybrid/rerank not significant. Values now
   in §4.3.
3. **Latency** — resolved. CPU timing added in §4.6: BM25 0.52 s/query, rerank 164.4 s/query;
   dense cold-start honestly reported as *not measured* (cache clear ineffective: p50 0.014 s
   is a cache hit). The production TalentLens pipeline latency (37s first query / 13s
   repeat, 100K corpus) remains a *separate system* and is not quoted in this paper.

---

## FILES CHANGED

- `PAPER_DRAFT.md` — v0.1 → v0.2: abstract updated; §3 n clarification + α spec;
  §4.1 post-fix H1 Δ CIs; §4.3 AP tables synced to significance.json p-values and
  α = 0.0021 (family = 24); §4.4 ES significance resolved; §4.5
  recommendation; §5 threat #8; Appendix A/B updated.

**needs human:** venue/author block and final paper claims
(per standing rule, only the user decides what the paper asserts).

**Resolved since v0.2:** latency (§4.6 added, dense cold-start marked unmeasured).
