# -*- coding: utf-8 -*-
import pytest
from src.llm_reranker import generate_rule_based_reasoning

def _make_candidate(title="AI Engineer", yoe=6.0, skills=None, trap_score=0.0):
    return {
        "candidate_id": "CAND_TEST_001",
        "profile": {
            "current_title": title,
            "current_company": "TechCorp",
            "years_of_experience": yoe
        },
        "skills": [{"name": s, "proficiency": "expert"} for s in (skills or ["Python", "NLP"])],
        "career_history": [{"title": title, "company": "TechCorp", "duration_months": 24}],
        "redrob_signals": {"recruiter_response_rate": 0.8, "notice_period_days": 30, "open_to_work_flag": True}
    }

def test_top_tier_match_banding():
    cand = _make_candidate()
    reasoning = generate_rule_based_reasoning(
        candidate=cand,
        score=92.5,
        breakdown={"trap_score": 0.0, "title_seniority_match": 0.9},
        target_skills={"Python", "NLP"}
    )
    assert reasoning.startswith("Top-tier match:")

def test_strong_match_banding():
    cand = _make_candidate()
    reasoning = generate_rule_based_reasoning(
        candidate=cand,
        score=75.0,
        breakdown={"trap_score": 0.0, "title_seniority_match": 0.8},
        target_skills={"Python", "NLP"}
    )
    assert reasoning.startswith("Strong match:")

def test_moderate_match_banding():
    cand = _make_candidate()
    reasoning = generate_rule_based_reasoning(
        candidate=cand,
        score=58.0,
        breakdown={"trap_score": 0.0, "title_seniority_match": 0.6},
        target_skills={"Python"}
    )
    assert reasoning.startswith("Moderate match:")

def test_weak_match_banding():
    cand = _make_candidate(title="Junior Support", yoe=1.0, skills=["Excel"])
    reasoning = generate_rule_based_reasoning(
        candidate=cand,
        score=38.0,
        breakdown={"trap_score": 0.0, "title_seniority_match": 0.2},
        target_skills={"Python", "NLP"}
    )
    assert reasoning.startswith("Weak match")

def test_trap_disqualification_reasoning():
    cand = _make_candidate(title="Lead Recruiter", yoe=10.0)
    cand["_trap_reason"] = "impossible career velocity"
    reasoning = generate_rule_based_reasoning(
        candidate=cand,
        score=99.0,
        breakdown={"trap_score": 0.85},
        target_skills={"Python"}
    )
    assert "Disqualified decoy profile" in reasoning