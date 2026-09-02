"""
Precomputation script to generate dense BGE embeddings for all candidates offline.
Saves candidate_embeddings.npy (100K x 768 float32 matrix) and candidate_ids.json.
"""
import os
import sys
import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project root to python path to allow running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import stream_candidates
from src.embedding_scorer import EmbeddingScorer
from src.candidate_text import build_candidate_embedding_text

def build_candidate_text(cand: dict) -> str:
    return build_candidate_embedding_text(cand)


def text_fingerprint(candidates, build_fn, model_name: str, max_seq: int) -> str:
    """Compute SHA-256 fingerprint over the actual embedded text (sampled every 500th candidate)."""
    h = hashlib.sha256()
    h.update(f"{model_name}|{max_seq}|{len(candidates)}".encode())
    for c in candidates[::500]:
        h.update(build_fn(c).encode("utf-8"))
    return h.hexdigest()


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

    # Write embeddings_meta.json with text fingerprint for cache validation
    meta_path = data_dir / "embeddings_meta.json"
    fingerprint = text_fingerprint(
        # Rebuild texts from stored candidates for fingerprinting.
        # We already have candidate_texts in memory from the loop above.
        # But candidate_texts is a list of strings, not candidates.
        # We need the original candidates. Since we streamed them, we re-read.
        # Actually, we can use the candidate_texts list directly.
        # text_fingerprint expects (candidates, build_fn, model_name, max_seq)
        # We'll adapt: just hash the texts directly.
        [], build_candidate_text, scorer.model_name, 160
    )
    # More efficient: hash the texts we already built
    h = hashlib.sha256()
    h.update(f"{scorer.model_name}|160|{len(candidate_texts)}".encode())
    for t in candidate_texts[::500]:
        h.update(t.encode("utf-8"))
    fingerprint = h.hexdigest()

    meta = {
        "text_fingerprint": fingerprint,
        "model_name": scorer.model_name,
        "max_seq_length": 160,
        "n_candidates": len(candidate_texts),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved embeddings metadata to {meta_path}")
    print(f"Text fingerprint: {fingerprint}")
        
    print("\n" + "=" * 60)
    print("Precomputation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
