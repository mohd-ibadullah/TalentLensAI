import os
import sys
import pytest

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_scorer import (
    match_skill,
    compute_skill_match_score,
    compute_title_seniority_match,
    compute_signal_bonus,
    calculate_candidate_score
)

def test_match_skill():
    # Exact match
    skill, is_match, score = match_skill("Python", ["Python", "Java"])
    assert is_match
    assert skill == "Python"
    assert score == 1.0

    # Substring match
    skill, is_match, score = match_skill("LLM Fine-tuning", ["Fine-tuning"])
    assert is_match
    assert skill == "Fine-tuning"
    assert score == 0.9

    # Fuzzy match
    skill, is_match, score = match_skill("PyTorche", ["PyTorch"])
    assert is_match
    assert skill == "PyTorch"
    assert score >= 0.85

    # No match
    _, is_match, _ = match_skill("Accounting", ["Python", "NLP"])
    assert not is_match

def test_compute_skill_match_score():
    parsed_jd = {
        "required_skills": ["Python", "NLP"],
        "nice_to_have_skills": ["PyTorch"]
    }
    
    # Matching all required
    candidate_skills = [
        {"name": "Python", "proficiency": "advanced"},
        {"name": "NLP", "proficiency": "advanced"}
    ]
    score = compute_skill_match_score(candidate_skills, parsed_jd)
    assert score >= 0.99  # capped at 0.994
    
    # Matching partial
    candidate_skills_partial = [
        {"name": "Python", "proficiency": "advanced"}
    ]
    score_partial = compute_skill_match_score(candidate_skills_partial, parsed_jd)
    assert 0.4 < score_partial < 0.6

def test_compute_title_seniority_match():
    parsed_jd = {
        "role_title": "Senior AI Engineer",
        "min_years_experience": 5.0
    }
    
    # Perfect title and YoE match
    profile = {
        "current_title": "Senior AI Engineer",
        "years_of_experience": 6.0
    }
    title_score = compute_title_seniority_match(profile, parsed_jd)
    assert title_score >= 0.90  # hard cap 0.97
    
    # Unrelated title, lower experience
    profile_weak = {
        "current_title": "Accountant",
        "years_of_experience": 2.5
    }
    score_weak = compute_title_seniority_match(profile_weak, parsed_jd)
    assert score_weak < 0.5

def test_compute_signal_bonus():
    # Base neutral signals
    signals = {}
    assert compute_signal_bonus(signals) == 0.5
    
    # High score signals + notice period bonus
    signals_good = {
        "profile_completeness_score": 90,
        "recruiter_response_rate": 0.80,
        "interview_completion_rate": 0.90,
        "notice_period_days": 15
    }
    score_good = compute_signal_bonus(signals_good)
    assert score_good > 0.8
    
    # Low response rate penalty
    signals_bad = {
        "profile_completeness_score": 40,
        "recruiter_response_rate": 0.05,
        "notice_period_days": 90
    }
    score_bad = compute_signal_bonus(signals_bad)
    assert score_bad < 0.4

def test_calculate_candidate_score():
    candidate = {
        "profile": {"current_title": "Senior AI Engineer", "years_of_experience": 6.0},
        "skills": [{"name": "Python"}, {"name": "NLP"}],
        "redrob_signals": {"profile_completeness_score": 90}
    }
    parsed_jd = {
        "role_title": "Senior AI Engineer",
        "required_skills": ["Python", "NLP"],
        "nice_to_have_skills": []
    }
    
    score, breakdown = calculate_candidate_score(
        candidate=candidate,
        semantic_similarity=0.90,
        trap_score=0.0,
        parsed_jd=parsed_jd
    )
    
    assert score > 80.0
    assert breakdown["trap_score"] == 0.0
    
    # Candidate with trap penalty
    score_trap, breakdown_trap = calculate_candidate_score(
        candidate=candidate,
        semantic_similarity=0.90,
        trap_score=1.0,
        parsed_jd=parsed_jd
    )
    assert score_trap < score
