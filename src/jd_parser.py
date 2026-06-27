import json
import re

# A default list of known skills to check against for free-text extraction
COMMON_SKILLS_TAXONOMY = [
    "Python", "NLP", "Vector Search", "Embeddings", "LLM Fine-tuning", "Fine-tuning LLMs",
    "Information Retrieval", "Ranking Systems", "PyTorch", "TensorFlow", "Distributed Systems",
    "MLOps", "RAG", "Milvus", "FAISS", "Pinecone", "SQL", "Spark", "Airflow", "Docker", "Kubernetes",
    "AWS", "GCP", "Azure", "Git", "Java", "C++", "Scala", "React", "TypeScript", "JavaScript",
    "HTML", "CSS", "Tailwind", "Pandas", "NumPy", "Scikit-learn", "Keras", "Hugging Face",
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision"
]

def parse_job_description(jd_input: dict | str) -> dict:
    """
    Parses a job description which can be either a dictionary (structured JSON) or a free-text string.
    Returns a structured dictionary of requirements.
    """
    if isinstance(jd_input, dict):
        return {
            "role_title": jd_input.get("role_title", "Senior AI Engineer"),
            "min_years_experience": float(jd_input.get("min_years_experience", 5.0)),
            "seniority_level": jd_input.get("seniority_level", "Senior"),
            "required_skills": [s.strip() for s in jd_input.get("required_skills", [])],
            "nice_to_have_skills": [s.strip() for s in jd_input.get("nice_to_have_skills", [])],
            "domain_keywords": [k.strip() for k in jd_input.get("domain_keywords", [])]
        }
    
    # If it is a string (free-text), extract using rules
    text = str(jd_input)
    text_lower = text.lower()
    
    # Extract years of experience
    min_years = 0.0
    exp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp)?', text_lower)
    if exp_matches:
        min_years = max(float(val) for val in exp_matches)
    else:
        # Fallback generic check
        if "senior" in text_lower:
            min_years = 4.0
        elif "lead" in text_lower or "principal" in text_lower:
            min_years = 7.0
            
    # Seniority Level
    seniority = "Mid"
    if "senior" in text_lower:
        seniority = "Senior"
    elif "lead" in text_lower:
        seniority = "Lead"
    elif "principal" in text_lower:
        seniority = "Principal"
    elif "junior" in text_lower:
        seniority = "Junior"
        
    # Extract role title (e.g. "Role: Senior AI Engineer" or similar, or first line)
    role_title = "AI Engineer"
    title_match = re.search(r'(?:role|title|position)\s*:\s*(.*)', text, re.IGNORECASE)
    if title_match:
        role_title = title_match.group(1).split('\n')[0].strip()
    else:
        # Try to find common titles
        for title in ["Senior AI Engineer", "Data Scientist", "ML Engineer", "Machine Learning Engineer", "Backend Engineer", "Frontend Engineer"]:
            if title.lower() in text_lower:
                role_title = title
                break
                
    # Extract skills by matching taxonomy
    extracted_skills = []
    for skill in COMMON_SKILLS_TAXONOMY:
        # Check for word-boundary matching
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        # Special handling for skills with hyphens or slashes
        if '-' in skill or '/' in skill:
            pattern = re.escape(skill.lower())
        if re.search(pattern, text_lower):
            extracted_skills.append(skill)
            
    # Separate required vs nice-to-have based on keywords in text
    required = []
    nice_to_have = []
    
    # Split text into sentences or sections to check context
    must_sections = []
    nice_sections = []
    
    parts = re.split(r'\n+', text_lower)
    current_section = "general"
    for part in parts:
        if "must-have" in part or "required" in part or "essential" in part:
            current_section = "required"
        elif "nice-to-have" in part or "preferred" in part or "plus" in part or "optional" in part:
            current_section = "nice"
            
        if current_section == "required":
            must_sections.append(part)
        elif current_section == "nice":
            nice_sections.append(part)
        else:
            must_sections.append(part) # default to required context
            
    must_text = " ".join(must_sections)
    nice_text = " ".join(nice_sections)
    
    for skill in extracted_skills:
        skill_lower = skill.lower()
        if skill_lower in nice_text and skill_lower not in must_text:
            nice_to_have.append(skill)
        else:
            required.append(skill)
            
    # Domain keywords are usually derived from title and skills
    domain_keywords = [role_title]
    if "ai" in text_lower or "artificial intelligence" in text_lower:
        domain_keywords.append("AI")
    if "ml" in text_lower or "machine learning" in text_lower:
        domain_keywords.append("ML")
    if "search" in text_lower:
        domain_keywords.append("Search")
    if "retrieval" in text_lower:
        domain_keywords.append("Retrieval")
    if "ranking" in text_lower:
        domain_keywords.append("Ranking")
        
    return {
        "role_title": role_title,
        "min_years_experience": min_years,
        "seniority_level": seniority,
        "required_skills": required,
        "nice_to_have_skills": nice_to_have,
        "domain_keywords": list(set(domain_keywords))
    }
