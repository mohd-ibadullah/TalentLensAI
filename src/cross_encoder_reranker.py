"""
Cross-Encoder Reranker for top-100 candidate precision improvement.
Uses cross-encoder/ms-marco-MiniLM-L6-v2 to score (JD, candidate) pairs
and blends with original feature scores.
"""
import time
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """
        Load a cross-encoder model for pairwise relevance scoring.
        ms-marco-MiniLM-L6-v2 is optimized for passage retrieval and runs fast on CPU.
        """
        print(f"Loading cross-encoder model '{model_name}' (CPU)...")
        self.model = CrossEncoder(model_name, max_length=256)
        self.model_name = model_name

    def build_candidate_text(self, candidate: dict) -> str:
        """Build a concise text representation of a candidate for cross-encoder input."""
        profile = candidate.get("profile", {})
        title = profile.get("current_title", "")
        summary = profile.get("summary", "")[:200]
        skills = candidate.get("skills", [])
        top_skills_str = ", ".join([s.get("name", "") for s in skills[:12]])
        yoe = profile.get("years_of_experience", 0)
        return f"{title} ({yoe} years). {summary}. Skills: {top_skills_str}"

    def rerank(self, jd_text: str, candidates: list[dict], blend_weight: float = 0.4, min_yoe: float = 0.0) -> list[dict]:
        """
        Rerank candidates using cross-encoder scores blended with original scores.
        
        Args:
            jd_text: The job description text for pairing.
            candidates: List of candidate dicts with '_final_score' already set.
            blend_weight: Weight for cross-encoder score (1 - blend_weight for original).
                          Default 0.4 means 60% original + 40% cross-encoder.
            min_yoe: Minimum years of experience requirement to enforce.
        
        Returns:
            Reranked list of candidates with updated '_final_score' and '_rank'.
        """
        if not candidates:
            return candidates

        print(f"Cross-encoder reranking {len(candidates)} candidates...")
        t_start = time.time()

        # Build input pairs: (jd_text, candidate_text)
        pairs = []
        for cand in candidates:
            cand_text = self.build_candidate_text(cand)
            pairs.append((jd_text, cand_text))

        # Score all pairs in one batch
        ce_scores = self.model.predict(pairs, show_progress_bar=False)

        # Normalize cross-encoder scores to [0, 100] range to match feature scores
        ce_min = float(min(ce_scores))
        ce_max = float(max(ce_scores))
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        for i, cand in enumerate(candidates):
            # Normalize CE score to [0, 100]
            normalized_ce = ((float(ce_scores[i]) - ce_min) / ce_range) * 100.0

            # Store original score before blending
            original_score = cand["_final_score"]
            cand["_original_feature_score"] = original_score
            cand["_cross_encoder_score"] = normalized_ce

            # Blend: 60% original feature score + 40% cross-encoder score
            blended = (1 - blend_weight) * original_score + blend_weight * normalized_ce
            
            # Apply YoE deficit penalty to blended score
            profile = cand.get("profile", {})
            yoe = float(profile.get("years_of_experience", 0.0))
            if min_yoe > 0.0:
                if yoe < 4.0:
                    blended -= 50.0
                elif yoe < min_yoe:
                    blended -= 15.0
                
            cand["_final_score"] = blended

        # Re-sort by blended score descending, tie-break by candidate_id ascending
        candidates.sort(key=lambda x: (-x["_final_score"], x["candidate_id"]))

        # Re-assign ranks
        for i, cand in enumerate(candidates):
            cand["_rank"] = i + 1

        duration = time.time() - t_start
        print(f"Cross-encoder reranking completed in {duration:.2f}s")

        return candidates
