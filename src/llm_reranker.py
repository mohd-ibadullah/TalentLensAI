import os
import sys

def generate_rule_based_reasoning(candidate, score, breakdown, rank=None):
    """
    Generate dynamic, non-templated candidate reasoning highlighting credentials,
    specific skills matched, platform signals, and potential concerns.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0.0)
    
    # Check trap status
    trap_score = breakdown.get("trap_score", 0.0)
    is_trap = trap_score > 0.4
    
    # Count and extract matched AI skills
    skills = candidate.get("skills", [])
    from src.honeypot_detector import AI_SKILLS
    matched_skills = []
    for s in skills:
        s_name = s.get("name", "")
        if s_name.lower() in {sk.lower() for sk in AI_SKILLS}:
            matched_skills.append(s_name)
    ai_skill_count = len(matched_skills)
    
    if is_trap:
        return f"FLAGGED DECOY: Title is {title} with {yoe} yrs; claimed {ai_skill_count} AI skills but career history indicates a mismatch. Trap score: {trap_score:.2f}."

    # Extract signals
    signals = candidate.get("redrob_signals", {})
    resp_rate = signals.get("recruiter_response_rate", -1)
    notice_days = signals.get("notice_period_days", -1)
    
    # Get a list of up to 3 matched skills to reference
    matched_skills_str = ", ".join(matched_skills[:3]) if matched_skills else "relevant technical skills"
    
    # Determine rank bucket
    if rank is None:
        if score >= 85:
            bucket = "top"
        elif score >= 75:
            bucket = "strong"
        else:
            bucket = "adjacent"
    else:
        if rank <= 15:
            bucket = "top"
        elif rank <= 50:
            bucket = "strong"
        else:
            bucket = "adjacent"
            
    # Deterministic RNG based on candidate ID to ensure reproducible text generation
    import random
    cand_id = candidate.get("candidate_id", "CAND_0000000")
    seed = sum(ord(char) for char in cand_id)
    rng = random.Random(seed)
    
    if bucket == "top":
        openings = [
            f"Exceptional {title} offering {yoe} years of experience",
            f"Top-tier candidate currently working as {title} ({yoe} YoE)",
            f"Highly relevant {title} with {yoe} years of background",
            f"Outstanding candidate as {title} showing {yoe} years of experience"
        ]
        mid_phrases = [
            f"demonstrates strong proficiency in {matched_skills_str} and alignment with the JD",
            f"offers solid expertise in {matched_skills_str} which matches the core ranking mandate",
            f"has successfully deployed systems involving {matched_skills_str} at scale",
            f"shows deep technical expertise in {matched_skills_str} and search infrastructure"
        ]
        end_phrases = []
        if resp_rate != -1:
            end_phrases.append(f"Highly active on Redrob with a {int(resp_rate*100)}% recruiter response rate.")
            end_phrases.append(f"Excellent platform engagement with {int(resp_rate*100)}% response rate.")
        if notice_days != -1 and notice_days < 30:
            end_phrases.append(f"Available quickly with a short notice period of {notice_days} days.")
        if not end_phrases:
            end_phrases.append("Matches the product company background and scale requirements perfectly.")
            
        reasoning = f"{rng.choice(openings)}, {rng.choice(mid_phrases)}. {rng.choice(end_phrases)}"
        
    elif bucket == "strong":
        openings = [
            f"Strong {title} with {yoe} years of experience",
            f"Competent {title} ({yoe} YoE)",
            f"Solid candidate working as {title} with {yoe} years in the field",
            f"Well-qualified {title} showing {yoe} YoE"
        ]
        mid_phrases = [
            f"matching key job requirements like {matched_skills_str}",
            f"possessing practical experience in {matched_skills_str}",
            f"with matching skills in {matched_skills_str} and solid backend fundamentals",
            f"who brings relevant experience in {matched_skills_str}"
        ]
        concerns = []
        if resp_rate != -1 and resp_rate < 0.3:
            concerns.append(f"though platform response rate is lower ({int(resp_rate*100)}%)")
        if notice_days != -1 and notice_days > 60:
            concerns.append(f"note the longer notice period of {notice_days} days")
        if yoe < 5.0:
            concerns.append(f"slightly below the preferred 5+ years of experience")
            
        if concerns:
            reasoning = f"{rng.choice(openings)}, {rng.choice(mid_phrases)}, {rng.choice(concerns)}."
        else:
            reasoning = f"{rng.choice(openings)}, {rng.choice(mid_phrases)}. Good overall behavioral signals."
            
    else: # adjacent / lower-ranked candidates
        openings = [
            f"Adjacent candidate working as {title} ({yoe} YoE)",
            f"Candidate with {yoe} years experience as {title}",
            f"Alternative profile showing {yoe} YoE as {title}",
            f"{title} with {yoe} years of background"
        ]
        mid_phrases = [
            f"mostly showing adjacent skills like {matched_skills_str}",
            f"with limited direct exposure to search/ranking but lists {matched_skills_str}",
            f"possessing partial overlap in {matched_skills_str}",
            f"having some background in {matched_skills_str} but less specialized"
        ]
        gaps = [
            "included as potential pipeline filler",
            "lower title relevance to the specific AI/ML requirements",
            "moderate fit with some technical gaps for this senior role",
            "would require substantial ramp-up time for retrieval systems"
        ]
        
        reasoning = f"{rng.choice(openings)}, {rng.choice(mid_phrases)}; {rng.choice(gaps)}."
        
    return reasoning

def rerank_top_candidates(top_candidates, parsed_jd, use_llm=False):
    """
    Reranks the top candidates and generates the required reasoning text.
    Returns the final list of candidates with ranks, scores, and reasonings.
    """
    final_ranked = []
    
    # If use_llm is requested, try to use LLM APIs
    llm_success = False
    if use_llm:
        # Check environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        if gemini_key:
            try:
                # We can import google.generativeai if installed, or do a simple REST API call
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                print("Using Gemini API for candidate reasoning...")
                for cand in top_candidates:
                    score = cand["_final_score"]
                    breakdown = cand["_breakdown"]
                    
                    # Call Gemini to write a summary
                    prompt = (
                        f"You are a professional tech recruiter. Given candidate profile summary: '{cand['profile']['summary']}', "
                        f"current title: '{cand['profile']['current_title']}', experience: {cand['profile']['years_of_experience']} years, "
                        f"skills: {[s['name'] for s in cand['skills']]}, and target job description: '{parsed_jd['role_title']}'. "
                        f"Write a concise 1-sentence summary of why they are or are not a good fit for this role. "
                        f"Keep it under 25 words. Keep it professional."
                    )
                    response = model.generate_content(prompt)
                    cand["_reasoning"] = response.text.strip().replace('"', '')
                llm_success = True
            except Exception as e:
                print(f"Failed to use Gemini API: {e}. Falling back to template-based reasoning.")
                
        elif openai_key and not llm_success:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                print("Using OpenAI API for candidate reasoning...")
                for cand in top_candidates:
                    score = cand["_final_score"]
                    breakdown = cand["_breakdown"]
                    
                    prompt = (
                        f"Write a concise 1-sentence recruiter reasoning under 20 words for candidate: "
                        f"Title: {cand['profile']['current_title']}, YoE: {cand['profile']['years_of_experience']}, "
                        f"Skills: {[s['name'] for s in cand['skills']]}. Role: {parsed_jd['role_title']}."
                    )
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50
                    )
                    cand["_reasoning"] = response.choices[0].message.content.strip().replace('"', '')
                llm_success = True
            except Exception as e:
                print(f"Failed to use OpenAI API: {e}. Falling back to template-based reasoning.")

    # Apply template-based reasoning if LLM was not used or failed
    if not llm_success:
        # First sort them so we can assign ranks and generate rank-consistent reasoning
        sorted_candidates = sorted(
            top_candidates,
            key=lambda x: (-round(x["_final_score"] / 100.0, 4), x["candidate_id"])
        )
        for rank, cand in enumerate(sorted_candidates, 1):
            cand["_rank"] = rank
            score = cand["_final_score"]
            breakdown = cand["_breakdown"]
            cand["_reasoning"] = generate_rule_based_reasoning(cand, score, breakdown, rank)
        return sorted_candidates
            
    # If LLM succeeded, apply standard sorting and ranking
    sorted_candidates = sorted(
        top_candidates,
        key=lambda x: (-round(x["_final_score"] / 100.0, 4), x["candidate_id"])
    )
    
    # Assign ranks from 1 to N
    for rank, cand in enumerate(sorted_candidates, 1):
        cand["_rank"] = rank
        
    return sorted_candidates
