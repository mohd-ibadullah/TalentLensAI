# -*- coding: utf-8 -*-
"""
Name-blindness tests: verify that candidate scoring is invariant to
changes in anonymized_name, candidate_id, and other non-skill fields.

These tests ensure the pipeline does not leak identity signals through
scoring — a prerequisite for fair candidate ranking.
"""
import copy
import pytest

from src.feature_scorer import (
    compute_skill_match_score,
    compute_title_seniority_match,
    compute_signal_bonus,
    calculate_candidate_score,
)
from src.honeypot_detector import detect_trap
from src.llm_reranker import generate_rule_based_reasoning

# ── Fixtures ──────────────────────────────────────────────────────────────────

JD = {
    "role_title": "Senior AI Engineer",
    "required_skills": ["Python", "NLP", "PyTorch"],
    "nice_to_have_skills": ["Embeddings", "FAISS"],
    "min_years_experience": 5.0,
    "seniority_level": "senior",
    "domain_keywords": ["AI", "Search"],
}

def _base_candidate(name: str = "Test User", cid: str = "CAND_NAME_TEST_001"):
    """Return a fully-formed candidate dict with all scoring-relevant fields."""
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": name,
            "current_title": "ML Engineer",
            "current_company": "Acme Corp",
            "years_of_experience": 7.0,
            "summary": "ML engineer with 7 years of experience in NLP and search.",
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 15, "duration_months": 60},
            {"name": "NLP", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
            {"name": "PyTorch", "proficiency": "intermediate", "endorsements": 8, "duration_months": 36},
        ],
        "career_history": [
            {"company": "Acme Corp", "title": "ML Engineer", "duration_months": 48, "description": "NLP and search systems."},
            {"company": "StartupX", "title": "Data Scientist", "duration_months": 36, "description": "ML pipelines."},
        ],
        "redrob_signals": {
            "profile_completeness_score": 85,
            "recruiter_response_rate": 0.65,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "github_activity_score": 42,
        },
    }


NAMES = [
    "Priya Sharma",
    "John Smith",
    "Chen Wei",
    "Fatima Al-Rashid",
    "Santiago García",
    "Ольга Петрова",
    "田中太郎",
    "Anonymous",
]


# ── Skill scoring invariance ───────────────────────────────────────────────────

class TestSkillScoreBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_skill_match_score_unchanged(self, name):
        cand = _base_candidate(name=name)
        score = compute_skill_match_score(cand["skills"], JD)
        # All candidates share identical skills → identical score
        expected = compute_skill_match_score(_base_candidate()["skills"], JD)
        assert score == expected, f"Name '{name}' changed skill score"

    @pytest.mark.parametrize("name", NAMES)
    def test_title_seniority_unchanged(self, name):
        cand = _base_candidate(name=name)
        score = compute_title_seniority_match(cand["profile"], JD)
        expected = compute_title_seniority_match(_base_candidate()["profile"], JD)
        assert score == expected, f"Name '{name}' changed title score"

    @pytest.mark.parametrize("name", NAMES)
    def test_signal_bonus_unchanged(self, name):
        cand = _base_candidate(name=name)
        score = compute_signal_bonus(cand["redrob_signals"])
        expected = compute_signal_bonus(_base_candidate()["redrob_signals"])
        assert score == expected, f"Name '{name}' changed signal score"


# ── Full score invariance ─────────────────────────────────────────────────────

class TestFinalScoreBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_final_score_unchanged(self, name):
        cand = _base_candidate(name=name)
        score_a, _ = calculate_candidate_score(cand, 0.85, 0.0, JD)
        score_b, _ = calculate_candidate_score(_base_candidate(), 0.85, 0.0, JD)
        assert score_a == score_b, f"Name '{name}' changed final score: {score_a} vs {score_b}"

    def test_different_ids_same_score(self):
        """Even with different candidate_ids, scores should be identical."""
        c1 = _base_candidate(cid="CAND_AAA")
        c2 = _base_candidate(cid="CAND_ZZZ")
        s1, _ = calculate_candidate_score(c1, 0.85, 0.0, JD)
        s2, _ = calculate_candidate_score(c2, 0.85, 0.0, JD)
        assert s1 == s2


# ── Honeypot detector invariance ──────────────────────────────────────────────

class TestTrapDetectorBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_trap_score_unchanged(self, name):
        cand = _base_candidate(name=name)
        trap_a, _ = detect_trap(cand)
        trap_b, _ = detect_trap(_base_candidate())
        assert trap_a == trap_b, f"Name '{name}' changed trap score"


# ── Reasoning generator invariance ────────────────────────────────────────────

class TestReasoningBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_reasoning_excludes_name(self, name):
        """Generated reasoning must never contain the anonymized_name."""
        cand = _base_candidate(name=name)
        cand["_trap_score"] = 0.0
        cand["_trap_reason"] = "clean"
        reasoning = generate_rule_based_reasoning(
            candidate=cand,
            score=85.0,
            breakdown={"trap_score": 0.0, "title_seniority_match": 0.85},
            target_skills={"Python", "NLP", "PyTorch"},
        )
        assert name not in reasoning, (
            f"Reasoning contains name '{name}': {reasoning}"
        )


# ── Score breakdown field audit ───────────────────────────────────────────────

class TestBreakdownNoNameFields:
    def test_breakdown_keys_are_pure(self):
        """Breakdown dict must only contain scoring component keys — no name/id leakage."""
        cand = _base_candidate()
        _, breakdown = calculate_candidate_score(cand, 0.85, 0.0, JD)
        allowed_keys = {
            "semantic_similarity", "skill_match_score", "title_seniority_match",
            "signal_bonus", "trap_score", "raw_positive_score",
            "trap_penalty_applied", "yoe_penalty_applied", "career_bonus_applied",
            "disqualifier_penalty_applied", "behavioral_adjustment",
            "trap_factor", "yoe_factor", "disq_factor",
        }
        unexpected = set(breakdown.keys()) - allowed_keys
        assert not unexpected, f"Unexpected breakdown keys: {unexpected}"
