import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

# Add current folder to path
sys.path.append(os.path.dirname(__file__))

from data_loader import load_sample_candidates, stream_candidates
from jd_parser import parse_job_description
from bm25_filter import BM25Filter
from honeypot_detector import detect_trap
from embedding_scorer import EmbeddingScorer
from feature_scorer import calculate_candidate_score
from llm_reranker import rerank_top_candidates
from cross_encoder_reranker import CrossEncoderReranker

def run_ranking_pipeline(candidates_path: str, jd_input: dict, out_csv_path: str, top_n: int = 100, use_llm: bool = False, weights: dict | None = None) -> list[dict]:
    """
    Executes the end-to-end TalentLens AI ranking pipeline:
    1. Parse Job Description.
    2. Hybrid Retrieval (BM25 + Dense Semantic Recall).
    3. Honeypot Trap Detection + Semantic Embedding Scoring.
    4. Weighted Feature Scoring.
    5. Cross-Encoder Reranker (top 100 precision boost).
    6. Reasoning Generation + CSV Output.
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
    
    # Build JD text for embedding — enriched with domain keywords and seniority
    jd_embedding_text = (
        f"{parsed_jd['role_title']}. "
        f"Required skills: {', '.join(parsed_jd['required_skills'])}. "
        f"Nice to have: {', '.join(parsed_jd['nice_to_have_skills'])}. "
        f"Domain: {', '.join(parsed_jd.get('domain_keywords', []))}. "
        f"Seniority: {parsed_jd.get('seniority_level', 'senior')}"
    )
    
    # Check if precomputed embeddings exist
    project_root = Path(__file__).resolve().parent.parent
    npy_path = project_root / "data" / "candidate_embeddings.npy"
    json_path = project_root / "data" / "candidate_ids.json"
    
    has_precomputed = npy_path.exists() and json_path.exists()
    
    # Load embedding model
    embedding_scorer = EmbeddingScorer()
    if has_precomputed:
        embedding_scorer.load_precomputed_embeddings(str(npy_path), str(json_path))
    
    # 2. Stage 1: Hybrid Retrieval Filter
    print("\nStep 2: Performing Hybrid Retrieval Stage...")
    
    is_jsonl = candidates_path.lower().endswith(".jsonl")
    candidates_for_deep_scoring = []
    
    if not is_jsonl:
        # Development mode: load all into memory (only 50 profiles)
        print("Loading development sample candidates...")
        all_candidates = load_sample_candidates(candidates_path)
        print(f"Loaded {len(all_candidates)} candidates.")
        
        # Apply BM25 filtering
        bm25_filter = BM25Filter(all_candidates)
        candidates_for_deep_scoring = bm25_filter.filter_candidates(parsed_jd, top_n=2000)
    else:
        # Production mode: single-pass loading
        print("Streaming JSONL (single pass) to load all candidates...")
        all_candidates = []
        for cand in stream_candidates(candidates_path):
            all_candidates.append(cand)
            
        print(f"Loaded {len(all_candidates)} profiles into memory.")
        
        # Run BM25 filter (Lexical Recall)
        print("Building BM25 index and performing Lexical search...")
        bm25_filter = BM25Filter(all_candidates)
        bm25_candidates = bm25_filter.filter_candidates(parsed_jd, top_n=1000)
        
        if has_precomputed:
            # Run Dense Vector search (Semantic Recall)
            print("Performing Dense Semantic vector search...")
            dense_results = embedding_scorer.search_similar_candidates(jd_embedding_text, top_n=1000)
            
            # Map candidate IDs to candidate objects for fast retrieval
            cand_map = {c["candidate_id"]: c for c in all_candidates}
            
            seen_ids = set()
            # Add BM25 recall candidates first to preserve order
            for cand in bm25_candidates:
                cid = cand["candidate_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    candidates_for_deep_scoring.append(cand)
                    
            # Add Dense recall candidates who are not already selected
            for cid, sim in dense_results:
                if cid not in seen_ids:
                    cand = cand_map.get(cid)
                    if cand:
                        seen_ids.add(cid)
                        candidates_for_deep_scoring.append(cand)
            print(f"Hybrid retrieval complete: {len(bm25_candidates)} BM25 + {len(dense_results)} Dense -> {len(candidates_for_deep_scoring)} unique candidates.")
        else:
            print("Precomputed vectors not found. Falling back to BM25-only retrieval.")
            for cand in bm25_candidates:
                candidates_for_deep_scoring.append(cand)
                
    print(f"Coarse filter completed. Evaluating {len(candidates_for_deep_scoring)} candidates.")
    
    # 3. Stage 2 & 3: Honeypot Detection & Feature Scoring
    print("\nStep 3: Evaluating Honeypot trap scores and Semantic embeddings...")
    
    # Calculate similarities (Fast lookup if precomputed, otherwise compute on the fly)
    print("Computing profile similarity scores...")
    similarities = []
    
    if has_precomputed and is_jsonl:
        print("Loading similarities from precomputed cache...")
        jd_embedding_vec = embedding_scorer.get_embeddings([jd_embedding_text], is_query=True)[0]
        for cand in candidates_for_deep_scoring:
            sim = embedding_scorer.get_candidate_similarity_by_id(cand["candidate_id"], jd_embedding_vec)
            similarities.append(sim)
    else:
        # Pre-calculate texts for embedding computation on the fly
        print("Computing embeddings dynamically...")
        candidate_texts = []
        for cand in candidates_for_deep_scoring:
            profile = cand.get("profile", {})
            title = profile.get("current_title", "")
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            skills_str = ", ".join([s.get("name", "") for s in cand.get("skills", [])[:15]])
            career_titles = " | ".join([r.get("title", "") for r in cand.get("career_history", [])[:5]])
            candidate_texts.append(f"Title: {title}. Headline: {headline}. Skills: {skills_str}. Career: {career_titles}. Summary: {summary}")
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
        
    # 4. Stage 4: Sort by feature score to get top candidates
    print("\nStep 4: Sorting by feature scores...")
    scored_candidates.sort(key=lambda x: (-x["_final_score"], x["candidate_id"]))
    top_subset = scored_candidates[:min(150, len(scored_candidates))]
    
    # 5. Stage 5: Cross-Encoder Reranker (precision boost on top candidates)
    print("\nStep 5: Running Cross-Encoder Reranker...")
    cross_encoder = CrossEncoderReranker()
    top_subset = cross_encoder.rerank(
        jd_embedding_text,
        top_subset,
        blend_weight=0.4,
        min_yoe=float(parsed_jd.get("min_years_experience", 0.0))
    )
    
    # 6. Stage 6: Reasoning Generation
    print("\nStep 6: Generating candidate reasonings...")
    reranked_top = rerank_top_candidates(top_subset, parsed_jd, use_llm=use_llm)
    
    # Final output
    final_output_list = reranked_top[:min(top_n, len(reranked_top))]
    
    # Write to CSV
    print(f"\nWriting results to {out_csv_path}...")
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
