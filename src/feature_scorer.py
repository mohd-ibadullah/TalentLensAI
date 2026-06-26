import numpy as np
from rapidfuzz import fuzz

# Default weights as specified in the PRD
SCORING_WEIGHTS = {
    "semantic_similarity": 0.35,
    "skill_match_score": 0.30,
    "title_seniority_match": 0.15,
    "signal_bonus": 0.10,
    "trap_penalty": 0.40
}

PROFICIENCY_MULTIPLIERS = {
    "beginner": 0.6,
    "intermediate": 0.8,
    "advanced": 1.0,
    "expert": 1.2
}

def match_skill(cand_skill_name: str, target_skills: list[str]) -> tuple[str | None, bool, float]:
    """
    Uses RapidFuzz to match candidate's skill against a list of target skills.
    Returns (matched_skill_name, is_match, score) where score is 0-1.
    """
    cand_skill_clean = cand_skill_name.lower().strip()
    
    for skill in target_skills:
        skill_clean = skill.lower().strip()
        
        # Direct equality or substring check
        if cand_skill_clean == skill_clean:
            return skill, True, 1.0
        if skill_clean in cand_skill_clean or cand_skill_clean in skill_clean:
            return skill, True, 0.9
            
        # Fuzzy match
        ratio = fuzz.token_sort_ratio(cand_skill_clean, skill_clean)
        if ratio >= 85:
            return skill, True, ratio / 100.0
            
    return None, False, 0.0

def compute_skill_match_score(candidate_skills: list[dict], parsed_jd: dict) -> float:
    """
    Calculates a normalized score in [0.0, 1.0] indicating skill fit.
    Weighs required skills higher than nice-to-have, and scales based on 
    proficiency and endorsements.
    """
    required = parsed_jd.get("required_skills", [])
    nice = parsed_jd.get("nice_to_have_skills", [])
    
    if not required:
        return 1.0 # Avoid division by zero
        
    matched_required = set()
    matched_nice = set()
    total_points = 0.0
    
    # Max possible points would be matching all required skills at 'advanced' level (1.0 weight) with no endorsements
    max_possible_points = len(required) * 1.0
    
    for s in candidate_skills:
        s_name = s.get("name", "")
        s_prof = s.get("proficiency", "intermediate").lower()
        s_end = s.get("endorsements", 0)
        s_dur = s.get("duration_months", 0) or 0
        
        # Base multiplier from proficiency
        prof_mult = PROFICIENCY_MULTIPLIERS.get(s_prof, 0.8)
        
        # Endorsements multiplier: slight bonus up to +20%
        end_mult = 1.0 + min(0.20, s_end / 50.0)
        
        # Skill match checks
        req_match, is_req, req_score = match_skill(s_name, required)
        if is_req and req_match not in matched_required:
            matched_required.add(req_match)
            total_points += 1.0 * req_score * prof_mult * end_mult
            continue
            
        nice_match, is_nice, nice_score = match_skill(s_name, nice)
        if is_nice and nice_match not in matched_nice:
            matched_nice.add(nice_match)
            total_points += 0.5 * nice_score * prof_mult * end_mult

    # Normalize to [0.0, 1.0] and cap at 1.0
    normalized_score = min(1.0, total_points / max_possible_points) if max_possible_points > 0 else 0.0
    return normalized_score

def compute_title_seniority_match(profile: dict, parsed_jd: dict) -> float:
    """
    Calculates title alignment and years of experience fit.
    Returns a score in [0.0, 1.0].
    """
    current_title = profile.get("current_title", "").lower()
    yoe = float(profile.get("years_of_experience", 0.0))
    
    target_title = parsed_jd.get("role_title", "").lower()
    min_yoe = float(parsed_jd.get("min_years_experience", 4.0))
    
    # 1. Title Relevance Score
    title_relevance = 0.1
    
    # Negative keywords — roles clearly unrelated to Search/AI/NLP JD
    negative_titles = [
        "computer vision", "frontend", "front-end", "front end", "android",
        "ios developer", "ui designer", "ux designer", "graphic designer",
        "customer support", "sales", "marketing", "hr manager",
        "accountant", "finance", "legal", "operations manager",
        "network admin", "system administrator", "devops"
    ]
    
    # Check for negative title match first
    if any(neg in current_title for neg in negative_titles):
        title_relevance = 0.05  # Heavy penalty for clearly irrelevant roles
    # Direct exact match or strong containment of keyword search / AI terms
    elif any(term in current_title for term in target_title.split()):
        title_relevance = 1.0
    else:
        ai_terms = ["ai", "ml", "machine learning", "nlp", "search", "retrieval",
                     "ranking", "data scientist", "deep learning", "applied scientist"]
        if any(term in current_title for term in ai_terms):
            title_relevance = 0.8
        elif "software engineer" in current_title or "backend engineer" in current_title or "developer" in current_title or "analyst" in current_title:
            title_relevance = 0.6
        
    # 2. Years of Experience Fit — stronger penalty for far below minimum
    yoe_score = 0.0
    if yoe >= min_yoe:
        yoe_score = 1.0
    elif yoe >= min_yoe * 0.8:
        # Within 80-100% of requirement (e.g., 4-5 YoE when 5 required) — slight penalty
        yoe_score = 0.8
    elif yoe >= min_yoe * 0.6:
        # Within 60-80% of requirement (e.g., 3-4 YoE when 5 required) — moderate penalty
        yoe_score = 0.5
    elif yoe > 0:
        # Below 60% of requirement — heavy penalty
        yoe_score = 0.2 * (yoe / min_yoe)
        
    # Combine: 50% title relevance, 50% experience relevance (increased YoE weight)
    return 0.5 * title_relevance + 0.5 * yoe_score

def compute_signal_bonus(signals: dict) -> float:
    """
    Normalize Redrob signals. -1 values are treated as missing/neutral,
    not penalized. Returns a score in [0.0, 1.0].
    """
    scores = []
    
    # Profile completeness: 0 to 100
    comp = signals.get("profile_completeness_score", -1)
    if comp != -1:
        scores.append(comp / 100.0)
        
    # Recruiter response rate: 0 to 1
    resp = signals.get("recruiter_response_rate", -1)
    if resp != -1:
        scores.append(resp)
        
    # Interview completion rate: 0 to 1
    intv = signals.get("interview_completion_rate", -1)
    if intv != -1:
        scores.append(intv)
        
    # GitHub activity score: 0 to 100
    git = signals.get("github_activity_score", -1)
    if git != -1:
        scores.append(git / 100.0)
        
    # Open to work flag
    open_to_work = signals.get("open_to_work_flag", None)
    if open_to_work is not None:
        scores.append(0.8 if open_to_work else 0.2)
        
    # If no data, return a neutral score
    if not scores:
        base_score = 0.5
    else:
        base_score = float(np.mean(scores))
        
    # Apply advanced behavioral signal envelopes
    modifier = 0.0
    
    # 1. Notice Period Bonus (+0.05 if notice period <= 30 days and not -1)
    notice = signals.get("notice_period_days", -1)
    if notice != -1 and notice <= 30:
        modifier += 0.05
        
    # 2. Recruiter Response Rate Penalty (-0.05 if response rate < 0.15 and not -1)
    if resp != -1 and resp < 0.15:
        modifier -= 0.05
        
    # 3. Inactivity Penalty (-0.03 if last_active_date > 180 days relative to 2026-06-16)
    last_active = signals.get("last_active_date", None)
    if last_active:
        try:
            from datetime import datetime, date
            ref_date = date(2026, 6, 16)
            active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            if (ref_date - active_date).days > 180:
                modifier -= 0.03
        except Exception:
            pass
            
    final_score = base_score + modifier
    return float(np.clip(final_score, 0.0, 1.0))

def calculate_candidate_score(candidate: dict, semantic_similarity: float, trap_score: float, parsed_jd: dict, weights: dict | None = None) -> tuple[float, dict]:
    """
    Combines all scoring components into a final score in [0.0, 100.0].
    Returns final score and a dictionary containing individual components.
    """
    if weights is None:
        weights = SCORING_WEIGHTS
        
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # Compute components
    skill_score = compute_skill_match_score(skills, parsed_jd)
    title_score = compute_title_seniority_match(profile, parsed_jd)
    signals_score = compute_signal_bonus(signals)
    
    # Calculate weighted positive sum
    positive_sum = (
        weights["semantic_similarity"] * semantic_similarity +
        weights["skill_match_score"] * skill_score +
        weights["title_seniority_match"] * title_score +
        weights["signal_bonus"] * signals_score
    )
    
    # Normalization factor for positive weights (sum of positive weights = 0.75 or 0.90 etc.)
    pos_weights_sum = (
        weights["semantic_similarity"] +
        weights["skill_match_score"] +
        weights["title_seniority_match"] +
        weights["signal_bonus"]
    )
    
    # Scale positive score out of 100
    scaled_positive = (positive_sum / pos_weights_sum) * 100.0 if pos_weights_sum > 0 else 0.0
    
    # Apply subtractive trap penalty (trap penalty is scaled out of 100 as well)
    penalty = weights["trap_penalty"] * trap_score * 100.0
    
    final_score = scaled_positive - penalty
    final_score = min(100.0, max(0.0, final_score))
    
    # Return score and component breakdown
    breakdown = {
        "semantic_similarity": float(semantic_similarity),
        "skill_match_score": float(skill_score),
        "title_seniority_match": float(title_score),
        "signal_bonus": float(signals_score),
        "trap_score": float(trap_score),
        "raw_positive_score": float(scaled_positive),
        "trap_penalty_applied": float(penalty)
    }
    
    return float(final_score), breakdown

