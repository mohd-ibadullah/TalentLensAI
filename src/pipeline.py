import os
import sys
import time
import json
import pandas as pd

# Add current folder to path
sys.path.append(os.path.dirname(__file__))

from data_loader import load_sample_candidates, stream_candidates
from jd_parser import parse_job_description
from bm25_filter import BM25Filter
from honeypot_detector import detect_trap
from embedding_scorer import EmbeddingScorer
from feature_scorer import calculate_candidate_score
from llm_reranker import rerank_top_candidates

def run_ranking_pipeline(candidates_path, jd_input, out_csv_path, top_n=100, use_llm=False, weights=None):
    """
    Executes the end-to-end TalentLens AI ranking pipeline:
    1. Parse Job Description.
    2. Coarse BM25 Filter (streams and selects top 3000).
    3. Run Honeypot Trap Detector and calculate feature scores.
    4. Optional LLM reranking and reasoning generation.
    5. Formats and writes the top 100 candidates to a validated CSV.
    """
    start_time = time.time()
    print("=" * 60)
    print("Starting TalentLens AI Ranking Pipeline")
    print(f"Candidates Path: {candidates_path}")
    print(f"Output CSV Path: {out_csv_path}")
    print("=" * 60)
    
    # 1. Parse Job Description
    print("Step 1: Parsing Job Description...")
    parsed_jd = parse_job_description(jd_input)
    print(f"Parsed Title: {parsed_jd['role_title']}")
    print(f"Required Skills: {parsed_jd['required_skills']}")
    print(f"Nice-to-have Skills: {parsed_jd['nice_to_have_skills']}")
    print(f"Min Years Exp: {parsed_jd['min_years_experience']}")
    
    # 2. Stage 1: Coarse Filtering
    print("\nStep 2: Performing Stage 1 Coarse Filtering...")
    
    # Handle sample JSON vs full JSONL
    is_jsonl = candidates_path.lower().endswith(".jsonl")
    
    candidates_for_deep_scoring = []
    
    if not is_jsonl:
        # Development mode: load all into memory (only 50 profiles)
        print("Loading development sample candidates...")
        all_candidates = load_sample_candidates(candidates_path)
        print(f"Loaded {len(all_candidates)} candidates.")
        
        # Apply BM25 filtering (keep all or top_n * 30 for testing)
        bm25_filter = BM25Filter(all_candidates)
        candidates_for_deep_scoring = bm25_filter.filter_candidates(parsed_jd, top_n=2000)
    else:
        # Production mode: single-pass loading
        print("Streaming JSONL (single pass) to load all candidates...")
        all_candidates = []
        for cand in stream_candidates(candidates_path):
            all_candidates.append(cand)
            
        print(f"Loaded {len(all_candidates)} profiles into memory.")
        
        # Run BM25 filter
        bm25_filter = BM25Filter(all_candidates)
        selected_candidates = bm25_filter.filter_candidates(parsed_jd, top_n=1500)
        
        for cand in selected_candidates:
            candidates_for_deep_scoring.append(cand)
            
    print(f"Coarse filter completed. Evaluating {len(candidates_for_deep_scoring)} candidates.")
    
    # 3. Stage 2 & 3: Honeypot Detection & Feature Scoring
    print("\nStep 3: Evaluating Honeypot trap scores and Semantic embeddings...")
    
    # Load embedding model
    embedding_scorer = EmbeddingScorer()
    
    # Pre-calculate candidate texts for embeddings
    # We embed candidate's key attributes (title, headline, skills, career) first, followed by summary.
    # This prevents critical structured details from being truncated.
    candidate_texts = []
    for cand in candidates_for_deep_scoring:
        profile = cand.get("profile", {})
        title = profile.get("current_title", "")
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        skills_str = ", ".join([s.get("name", "") for s in cand.get("skills", [])[:15]])
        career_titles = " | ".join([r.get("title", "") for r in cand.get("career_history", [])[:5]])
        candidate_texts.append(f"Title: {title}. Headline: {headline}. Skills: {skills_str}. Career: {career_titles}. Summary: {summary}")
        
    # Build JD text for embedding — enriched with domain keywords and seniority
    jd_embedding_text = (
        f"{parsed_jd['role_title']}. "
        f"Required skills: {', '.join(parsed_jd['required_skills'])}. "
        f"Nice to have: {', '.join(parsed_jd['nice_to_have_skills'])}. "
        f"Domain: {', '.join(parsed_jd.get('domain_keywords', []))}. "
        f"Seniority: {parsed_jd.get('seniority_level', 'senior')}"
    )
    
    print("Computing profile embeddings...")
    similarities = embedding_scorer.compute_similarity(jd_embedding_text, candidate_texts)
    
    print("Scoring candidates...")
    scored_candidates = []
    
    for cand, sim in zip(candidates_for_deep_scoring, similarities):
        # Stage 2: Trap Detection
        trap_score, trap_reason = detect_trap(cand)
        cand["_trap_score"] = trap_score
        cand["_trap_reason"] = trap_reason
        
        # Stage 3: Calculate Feature Score
        final_score, breakdown = calculate_candidate_score(cand, sim, trap_score, parsed_jd, weights=weights)
        
        cand["_final_score"] = final_score
        cand["_breakdown"] = breakdown
        scored_candidates.append(cand)
        
    # 4. Stage 4: LLM Rerank / Reasoning Generation (Runs on top 150 candidates only)
    print("\nStep 4: Running Stage 4 Reranking and Reasoning Generator...")
    # Sort by score descending to get top 150
    scored_candidates.sort(key=lambda x: (-x["_final_score"], x["candidate_id"]))
    top_subset_for_rerank = scored_candidates[:min(150, len(scored_candidates))]
    
    reranked_top = rerank_top_candidates(top_subset_for_rerank, parsed_jd, use_llm=use_llm)
    
    # Update candidate entries
    final_output_list = reranked_top[:min(top_n, len(reranked_top))]
    
    # 5. Output top candidates to CSV
    print(f"\nStep 5: Writing results to {out_csv_path}...")
    output_rows = []
    for cand in final_output_list:
        output_rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["_rank"],
            "score": round(cand["_final_score"] / 100.0, 4), # Convert back to [0.0, 1.0] float for submission schema
            "reasoning": cand["_reasoning"]
        })
        
    df_out = pd.DataFrame(output_rows)
    # Ensure columns match required schema
    df_out = df_out[["candidate_id", "rank", "score", "reasoning"]]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    
    duration = time.time() - start_time
    print(f"Pipeline completed successfully in {duration:.2f} seconds!")
    print("=" * 60)
    
    return final_output_list
