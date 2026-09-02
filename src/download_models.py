"""
Helper script to pre-download and cache embedding and cross-encoder models.
Run this script while you have an active network connection so that the ranking
pipeline can run completely offline during Stage 3 sandbox verification.
"""
import sys
import os

def main():
    print("=" * 60)
    print("Pre-downloading NLP models for TalentLens AI...")
    print("=" * 60)
    
    # 1. Download BAAI/bge-base-en-v1.5
    print("\n1. Pre-downloading BAAI/bge-base-en-v1.5...")
    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
        print("[OK] BAAI/bge-base-en-v1.5 downloaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error downloading BGE-base: {e}")
        sys.exit(1)
        
    # 2. Download cross-encoder/ettin-reranker-17m-v1
    print("\n2. Pre-downloading cross-encoder/ettin-reranker-17m-v1...")
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder("cross-encoder/ettin-reranker-17m-v1")
        print("[OK] cross-encoder/ettin-reranker-17m-v1 downloaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error downloading Cross-Encoder: {e}")
        sys.exit(1)
    
    # 3. Download ettin-reranker-17m (strictly dominant swap)
    print("\n3. Pre-downloading cross-encoder/ettin-reranker-17m...")
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder("cross-encoder/ettin-reranker-17m")
        print("[OK] cross-encoder/ettin-reranker-17m downloaded successfully!")
    except Exception as e:
        print(f"[WARN] Could not download Ettin reranker: {e}")
        
    print("\n" + "=" * 60)
    print("All models successfully cached! Pipeline is ready to run offline.")
    print("=" * 60)

if __name__ == "__main__":
    main()
