"""
Precomputation script to generate dense BGE embeddings for all candidates offline.
Saves candidate_embeddings.npy (100K x 768 float32 matrix) and candidate_ids.json.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to python path to allow running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import stream_candidates
from src.embedding_scorer import EmbeddingScorer

def build_candidate_text(cand: dict) -> str:
    """
    Constructs the exact same candidate text format used during ranking
    to ensure perfect query-passage alignment.
    """
    profile = cand.get("profile", {})
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    skills_str = ", ".join([s.get("name", "") for s in cand.get("skills", [])[:15]])
    career_titles = " | ".join([r.get("title", "") for r in cand.get("career_history", [])[:5]])
    return f"Title: {title}. Headline: {headline}. Skills: {skills_str}. Career: {career_titles}. Summary: {summary}"

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Try multiple paths for candidates.jsonl
    candidates_paths = [
        project_root.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
        project_root.parent / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
        project_root / "candidates.jsonl"
    ]
    
    candidates_path = None
    for p in candidates_paths:
        if p.exists():
            candidates_path = p
            break
            
    if not candidates_path:
        print("Error: Could not find candidates.jsonl in default locations.")
        sys.exit(1)
        
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    embeddings_out = data_dir / "candidate_embeddings.npy"
    ids_out = data_dir / "candidate_ids.json"
    
    print("=" * 60)
    print("TalentLens AI — Candidate Embedding Precomputation")
    print(f"Reading candidates from: {candidates_path}")
    print(f"Outputs will be saved to: {data_dir}")
    print("=" * 60)
    
    # Initialize scorer
    scorer = EmbeddingScorer()
    
    print("\nStreaming candidates and building texts...")
    candidate_ids = []
    candidate_texts = []
    
    count = 0
    for cand in stream_candidates(str(candidates_path)):
        candidate_ids.append(cand["candidate_id"])
        candidate_texts.append(build_candidate_text(cand))
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} profiles...")
            
    print(f"Finished reading {count} candidates.")
    
    print("\nComputing embeddings in batches (this will take 10-15 minutes on CPU)...")
    # Generate embeddings (is_query=False ensures no query instruction prefix is added)
    embeddings = scorer.get_embeddings(candidate_texts, batch_size=256, is_query=False)
    
    print(f"\nComputed matrix shape: {embeddings.shape}")
    print(f"Saving embeddings matrix to {embeddings_out}...")
    np.save(str(embeddings_out), embeddings)
    
    print(f"Saving candidate ID ordering to {ids_out}...")
    with open(ids_out, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
        
    print("\n" + "=" * 60)
    print("Precomputation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
