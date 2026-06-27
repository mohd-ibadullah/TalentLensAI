import numpy as np
from rapidfuzz import fuzz

# Default weights as specified in the PRD
SCORING_WEIGHTS = {
    "semantic_similarity": 0.40,
    "skill_match_score": 0.20,
    "title_seniority_match": 0.20,
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

    # Normalize to [0.0, 1.0] based on required skills count, cap at 0.994
    normalized_score = min(0.994, total_points / max_possible_points) if max_possible_points > 0 else 0.0
    return normalized_score

def compute_title_seniority_match(profile: dict, parsed_jd: dict) -> float:
    """
    Calculates title alignment and years of experience fit.
    3-component scoring: seniority match (40%), domain match (30%), YoE match (30%).
    Returns a score in [0.0, 0.97] — hard cap, never 100%.
    """
    current_title = profile.get("current_title", "").lower()
    yoe = float(profile.get("years_of_experience", 0.0))
    
    min_yoe = float(parsed_jd.get("min_years_experience", 5.0))
    jd_seniority = parsed_jd.get("seniority_level", "senior").lower()
    
    # 1. Seniority match (40% weight)
    senior_keywords = ["senior", "lead", "staff", "principal", "head", "director"]
    mid_keywords = ["engineer", "scientist", "analyst", "specialist", "developer"]
    
    if jd_seniority == "senior":
        if any(k in current_title for k in senior_keywords):
            title_score = 1.0
        elif any(k in current_title for k in mid_keywords):
            title_score = 0.72
        else:
            title_score = 0.40
    else:
        title_score = 0.80

    # 2. Domain match (30% weight) — is the title in AI/ML/Search domain?
    ai_keywords = ["ai", "ml", "nlp", "data", "machine learning", "search",
                   "recommendation", "applied scientist", "research", "deep learning"]
    
    # Negative domain — clearly non-tech or non-AI roles
    negative_domains = [
        "mechanical", "civil", "electrical", "chemical", "structural",
        "frontend", "android", "ios", "ui designer", "ux designer",
        "customer support", "sales", "marketing", "hr", "accountant",
        "finance", "legal", "operations", "graphic designer",
        "content writer", "teacher", "professor"
    ]
    
    if any(k in current_title for k in negative_domains):
        domain_score = 0.05
    elif any(k in current_title for k in ai_keywords):
        domain_score = 1.0
    elif "software" in current_title or "backend" in current_title or "cloud" in current_title:
        domain_score = 0.50
    else:
        domain_score = 0.30

    # 3. YoE match (30% weight) — continuous scoring
    if yoe >= min_yoe:
        yoe_score = min(1.0, 0.75 + (yoe - min_yoe) * 0.025)
    elif yoe > 0:
        yoe_score = max(0.0, (yoe / min_yoe) * 0.75)
    else:
        yoe_score = 0.0

    raw = (title_score * 0.40) + (domain_score * 0.30) + (yoe_score * 0.30)
    return min(raw, 0.97)  # Hard cap at 97%, never 100%

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
    
    # Calculate YoE deficit penalty (hard gate rules)
    yoe = float(profile.get("years_of_experience", 0.0))
    min_yoe = float(parsed_jd.get("min_years_experience", 5.0))
    
    # Calculate Career Relevance Bonus
    career_history = candidate.get("career_history", [])
    career_bonus = 0.0
    career_keywords = ["ranking", "retrieval", "recommendation", "search", "embedding", "recruiter", "product company"]
    
    matches_found = 0
    for job in career_history:
        job_title = job.get("title", "").lower()
        job_desc = job.get("description", "").lower() if job.get("description") else ""
        for kw in career_keywords:
            if kw in job_title or kw in job_desc:
                matches_found += 1
                
    if matches_found > 0:
        career_bonus = min(10.0, 3.0 * matches_found)
        
    # Calculate JD Disqualifiers (penalty)
    disqualifier_penalty = 0.0
    
    # A. Consulting-heavy career (TCS/Infosys/Wipro type)
    consulting_companies = [
        "tcs", "tata consultancy", "infosys", "wipro", "cognizant", 
        "accenture", "hcl", "tech mahindra", "capgemini", "lnt infotech", "l&t infotech"
    ]
    
    consulting_duration = 0.0
    total_duration = 0.0
    for job in career_history:
        comp = job.get("company", "").lower()
        dur = float(job.get("duration_months", 0.0) or 0.0)
        if comp:
            total_duration += dur
            if any(consulting in comp for consulting in consulting_companies):
                consulting_duration += dur
                
    is_consulting_heavy = (total_duration > 0 and (consulting_duration / total_duration) > 0.60)
    if is_consulting_heavy:
        disqualifier_penalty += 20.0
        
    # B. Pure CV / speech / robotics without NLP/IR
    cv_keywords = ["computer vision", "cv engineer", "speech recognition", "speech processing", "robotics", "audio engineer", "speech engineer", "image processing"]
    nlp_ir_keywords = ["nlp", "natural language", "search", "retrieval", "recommendation", "information retrieval", "ranking", "llm", "embeddings"]
    
    summary_lower = profile.get("summary", "").lower()
    title_lower = profile.get("current_title", "").lower()
    skills_lower = [s.get("name", "").lower() for s in skills]
    
    has_cv = any(kw in summary_lower or kw in title_lower or any(kw in sk for sk in skills_lower) for kw in cv_keywords)
    has_nlp_ir = any(kw in summary_lower or kw in title_lower or any(kw in sk for sk in skills_lower) for kw in nlp_ir_keywords)
    
    is_pure_cv = (has_cv and not has_nlp_ir)
    if is_pure_cv:
        disqualifier_penalty += 20.0
        
    # C. Marketing/HR title + AI skills stuffing
    hr_mktg_keywords = ["marketing", "hr ", "hr manager", "human resources", "recruiter", "sales"]
    is_hr_mktg_stuffing = any(kw in title_lower for kw in hr_mktg_keywords) and has_nlp_ir
    if is_hr_mktg_stuffing:
        disqualifier_penalty += 20.0
        
    # D. LangChain-only profile, no production ML and no ML titles
    has_langchain = "langchain" in summary_lower or any("langchain" in sk for sk in skills_lower)
    production_ml_keywords = ["pytorch", "tensorflow", "scikit-learn", "sklearn", "keras", "mlops", "production", "kubernetes", "docker", "aws", "gcp", "azure"]
    has_production_ml = any(kw in summary_lower or any(kw in sk for sk in skills_lower) for kw in production_ml_keywords)
    
    has_ml_title = False
    for job in career_history:
        job_title = job.get("title", "").lower()
        if any(term in job_title for term in ["machine learning", "ml ", "ai ", "nlp", "data scientist", "deep learning", "applied scientist"]):
            has_ml_title = True
            
    is_langchain_only = (has_langchain and not has_production_ml and not has_ml_title)
    if is_langchain_only:
        disqualifier_penalty += 20.0
        
    # E. Non-tech engineer (mechanical, civil, electrical, etc.) with zero AI/ML skills
    non_tech_engineer_titles = [
        "mechanical engineer", "civil engineer", "electrical engineer",
        "chemical engineer", "structural engineer", "environmental engineer",
        "industrial engineer"
    ]
    is_non_tech_engineer = any(nt in title_lower for nt in non_tech_engineer_titles)
    if is_non_tech_engineer and not has_nlp_ir and not has_production_ml:
        disqualifier_penalty += 100.0  # Complete rejection
        
    # Calculate Behavioral Signals adjustment
    behavioral_adjustment = 0.0
    resp = signals.get("recruiter_response_rate", -1)
    # Low response rate is already reflected in signal_bonus; avoid double-penalizing here.
    last_active = signals.get("last_active_date", None)
    if last_active:
        try:
            from datetime import datetime, date
            ref_date = date(2026, 6, 16)
            active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            if (ref_date - active_date).days > 180:
                behavioral_adjustment -= 10.0  # Increased inactivity penalty to -10.0
        except Exception:
            pass
            
    notice = signals.get("notice_period_days", -1)
    if notice != -1 and notice > 60:
        behavioral_adjustment -= 5.0
        
    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work and resp >= 0.70:
        behavioral_adjustment += 5.0
        
    # Multiplicative Veto Factors
    # 1. Honeypot/Trap Factor (Score set to exactly 0.0 if flagged)
    trap_factor = 1.0
    if trap_score >= 0.40:
        trap_factor = 0.0
        
    # 2. YoE Hard Gate — align with JD minimum (default 5.0 years)
    yoe_factor = 1.0
    yoe_penalty_applied = 0.0
    if yoe < min_yoe:
        yoe_factor = 0.0
        yoe_penalty_applied = 100.0
        
    # 3. Disqualifier Factors
    disq_factor = 1.0
    disqualifier_penalty_applied = 0.0
    if is_consulting_heavy:
        disq_factor *= 0.50
        disqualifier_penalty_applied += 20.0
    if is_pure_cv:
        disq_factor *= 0.50
        disqualifier_penalty_applied += 20.0
    if is_hr_mktg_stuffing:
        disq_factor *= 0.0  # Absolute rejection for marketing/HR stuffers
        disqualifier_penalty_applied += 100.0
    if is_langchain_only:
        disq_factor *= 0.50
        disqualifier_penalty_applied += 20.0
    if is_non_tech_engineer and not has_nlp_ir and not has_production_ml:
        disq_factor *= 0.0  # Absolute rejection for non-tech engineers
        disqualifier_penalty_applied += 100.0
        
    # Calculate multiplied score
    base_score = scaled_positive + career_bonus
    multiplied_score = base_score * trap_factor * yoe_factor * disq_factor
    
    # Final Score with behavioral adjustment
    final_score = multiplied_score + behavioral_adjustment
    # Cap score to avoid displaying a flat 100.0%
    final_score = min(99.4, max(0.0, final_score))
    
    # Return score and component breakdown (compatible with UI display)
    breakdown = {
        "semantic_similarity": float(semantic_similarity),
        "skill_match_score": float(skill_score),
        "title_seniority_match": float(title_score),
        "signal_bonus": float(signals_score),
        "trap_score": float(trap_score),
        "raw_positive_score": float(scaled_positive),
        "trap_penalty_applied": float(penalty),
        "yoe_penalty_applied": float(yoe_penalty_applied),
        "career_bonus_applied": float(career_bonus),
        "disqualifier_penalty_applied": float(disqualifier_penalty_applied),
        "behavioral_adjustment": float(behavioral_adjustment),
        "trap_factor": float(trap_factor),
        "yoe_factor": float(yoe_factor),
        "disq_factor": float(disq_factor)
    }
    
    return float(final_score), breakdown

