import re

# Set of AI/ML skills to check for keyword stuffing
AI_SKILLS = {
    "nlp", "vector search", "embeddings", "llm fine-tuning", "fine-tuning llms", 
    "information retrieval", "ranking systems", "pytorch", "tensorflow", "mlops", 
    "rag", "milvus", "faiss", "pinecone", "image classification", "object detection", 
    "speech recognition", "tts", "lora", "gans", "bentoml", "weights & biases", 
    "machine learning", "deep learning", "computer vision", "generative ai", 
    "statistical modeling", "apache flink", "databricks", "neural networks"
}

# Mapping of known template descriptions to the category they actually describe
BOILERPLATE_MAP = {
    "customer support team lead at a saas product": "Support/Operations",
    "mechanical engineering design role at a hardware-product company": "Mechanical Engineering",
    "content writing and seo strategy for a tech-focused publication": "Content/Marketing",
    "brand design and creative direction at a consumer-products company": "Graphic Design",
    "business analyst at a consulting firm": "Business Analyst",
    "senior accounting role at a mid-sized company": "Accounting",
    "operations management role at a logistics company": "Logistics/Operations"
}

# Mapping of keywords in titles to standard job categories
TITLE_CATEGORY_MAP = {
    "ai": "AI/ML", "ml": "AI/ML", "machine learning": "AI/ML", "nlp": "AI/ML", 
    "deep learning": "AI/ML", "computer vision": "AI/ML", "data scientist": "AI/ML", 
    "data science": "AI/ML", "search": "AI/ML", "information retrieval": "AI/ML",
    "retrieval": "AI/ML", "ranking": "AI/ML", "recommendation": "AI/ML",
    "accountant": "Accounting", "accounting": "Accounting", "finance": "Accounting", "tax": "Accounting",
    "hr": "HR/Recruiting", "human resources": "HR/Recruiting", "recruiter": "HR/Recruiting",
    "marketing": "Marketing/Content", "content writer": "Marketing/Content", "seo": "Marketing/Content",
    "writer": "Marketing/Content", "copywriter": "Marketing/Content",
    "operations": "Operations", "logistics": "Operations", "support": "Support", "customer": "Support",
    "mechanical": "Mechanical Engineering", "civil": "Civil Engineering",
    "graphic designer": "Graphic Design", "designer": "Graphic Design", "brand designer": "Graphic Design",
    "sales": "Sales", "account executive": "Sales", "business development": "Sales",
    "project manager": "Project Management", "program manager": "Project Management",
    "backend": "Software Engineering", "frontend": "Software Engineering", "fullstack": "Software Engineering",
    "software engineer": "Software Engineering", "developer": "Software Engineering", "coder": "Software Engineering"
}

def classify_title(title: str) -> str:
    """Classify a job title into a standard category."""
    if not title:
        return "Unknown"
    title_lower = title.lower()
    
    # Check for direct keyword matches in order of priority
    for kw, cat in TITLE_CATEGORY_MAP.items():
        # Word boundary or simple containment check
        if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
            return cat
        if kw in title_lower:
            return cat
            
    return "Other"

def check_boilerplate_description(desc: str) -> str | None:
    """Check if the description matches one of the known duplicate templates."""
    if not desc:
        return None
    desc_lower = desc.lower()
    for phrase, category in BOILERPLATE_MAP.items():
        if phrase in desc_lower:
            return category
    return None

def detect_trap(candidate: dict) -> tuple[float, str]:
    """
    Evaluates a candidate profile to determine if it is a decoy/honeypot.
    Returns:
        trap_score: float in [0.0, 1.0] where 1.0 is a confirmed decoy
        trap_reason: string detailing the reasons for the score
    """
    candidate_id = candidate.get("candidate_id", "Unknown")
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    
    current_category = classify_title(current_title)
    
    # 1. Check for AI Curiosity Mismatch Summary Boilerplate
    # The common honeypot summary contains: "My professional background is in marketing manager... Lately I've been curious..."
    is_decoy_summary = False
    summary_lower = summary.lower()
    if "marketing manager" in summary_lower and current_category != "Marketing/Content":
        is_decoy_summary = True
    if "curious about how ai tools could augment my work" in summary_lower:
        is_decoy_summary = True
    if "experimented with chatgpt" in summary_lower:
        is_decoy_summary = True
        
    # 2. Count AI/ML Skills Stuffing
    candidate_ai_skills = []
    for s in skills:
        s_name = s.get("name", "").lower()
        if s_name in AI_SKILLS or any(ai_s in s_name for ai_s in ["nlp", "vector search", "llm", "pytorch", "tensorflow", "machine learning"]):
            candidate_ai_skills.append(s.get("name"))
            
    ai_skill_count = len(candidate_ai_skills)
    
    # 3. Check Career History Title-Description Inconsistency
    mismatched_roles_count = 0
    boilerplate_roles_count = 0
    total_roles = len(career_history)
    
    for role in career_history:
        role_title = role.get("title", "")
        role_desc = role.get("description", "")
        
        # Check if description uses a known boilerplate template
        boilerplate_category = check_boilerplate_description(role_desc)
        role_title_category = classify_title(role_title)
        
        if boilerplate_category:
            boilerplate_roles_count += 1
            # Check if what the description actually describes matches the role title category
            if boilerplate_category != role_title_category:
                mismatched_roles_count += 1
                
    # 4. Compute Trap Score Components
    trap_score = 0.0
    reasons = []
    
    # If it is the classic marketing manager summary decoy
    if is_decoy_summary:
        trap_score += 0.4
        reasons.append("Generic summary template detected ('marketing manager' or 'AI tools curiosity' boilerplate)")
        
    # Title mismatch with AI skills stuffing
    # If the title is completely non-AI (e.g. HR, Accountant, Support, Mechanical) but they list multiple AI skills
    is_non_ai_title = current_category not in ["AI/ML", "Software Engineering", "Other", "Unknown"]
    if is_non_ai_title and ai_skill_count >= 3:
        # Heavily penalize
        skill_stuffing_score = min(0.4, 0.1 * ai_skill_count)
        trap_score += skill_stuffing_score
        reasons.append(f"Non-AI Title '{current_title}' ({current_category}) but stuffed with {ai_skill_count} AI/ML skills: {candidate_ai_skills}")
    elif is_non_ai_title and ai_skill_count >= 1:
        trap_score += 0.1
        
    # Career history inconsistencies
    if total_roles > 0:
        mismatch_ratio = mismatched_roles_count / total_roles
        if mismatch_ratio > 0.0:
            trap_score += 0.3 * mismatch_ratio
            reasons.append(f"Career history mismatch: {mismatched_roles_count} of {total_roles} roles have descriptions inconsistent with their titles (e.g. template descriptions reused)")
            
    # Normalize/cap trap score to [0.0, 1.0]
    trap_score = min(1.0, max(0.0, trap_score))
    
    if trap_score == 0:
        trap_reason = "Consistent profile: title, career history, and skills are aligned."
    else:
        trap_reason = " | ".join(reasons)
        
    return trap_score, trap_reason
