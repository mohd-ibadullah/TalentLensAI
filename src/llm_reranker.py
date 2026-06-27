import os
import sys

from src.feature_scorer import match_skill


def _matched_jd_skills(candidate: dict, target_skills: set[str]) -> list[str]:
    """Return JD skills matched via exact, substring, or fuzzy match."""
    required = list(target_skills)
    matched: list[str] = []
    seen: set[str] = set()
    for s in candidate.get("skills", []):
        name = s.get("name", "")
        if not name:
            continue
        hit, is_match, _ = match_skill(name, required)
        if is_match and hit and hit.lower() not in seen:
            matched.append(hit)
            seen.add(hit.lower())
    return matched


def generate_rule_based_reasoning(
    candidate: dict,
    score: float,
    breakdown: dict,
    target_skills: set[str] | None = None,
    rank: int | None = None,
) -> str:
    """
    Profile-specific recruiter reasoning for Stage 4 manual review.
    Uses facts from the candidate record — no generic template pools.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0.0)
    trap_score = breakdown.get("trap_score", 0.0)

    if trap_score >= 0.40:
        return (
            f"Disqualified decoy profile: {title} ({yoe} YoE) shows trap score {trap_score:.2f} — "
            f"{candidate.get('_trap_reason', 'title/skills/career mismatch')}."
        )

    skills = _matched_jd_skills(candidate, target_skills or set())
    skills_text = ", ".join(skills[:4]) if skills else "limited direct JD skill overlap"

    signals = candidate.get("redrob_signals", {})
    resp = signals.get("recruiter_response_rate", -1)
    notice = signals.get("notice_period_days", -1)
    open_work = signals.get("open_to_work_flag", None)

    signal_bits: list[str] = []
    if resp != -1:
        signal_bits.append(f"{int(resp * 100)}% recruiter response rate")
    if notice != -1:
        signal_bits.append(f"{notice}-day notice period")
    if open_work is True:
        signal_bits.append("open to work")
    signals_text = "; ".join(signal_bits) if signal_bits else "neutral platform signals"

    career = candidate.get("career_history", [])
    recent_role = career[0].get("title", "") if career else ""
    recent_company = career[0].get("company", "") if career else ""

    concerns: list[str] = []
    if yoe < 5.0:
        concerns.append(f"below JD minimum of 5 years (has {yoe})")
    if resp != -1 and resp < 0.20:
        concerns.append("low recruiter engagement")
    if notice != -1 and notice > 60:
        concerns.append(f"long notice ({notice} days)")
    if breakdown.get("title_seniority_match", 0) < 0.5:
        concerns.append("title domain is weak for senior AI/search role")
    if not skills:
        concerns.append("few explicit JD skills on profile")

    rank = rank or 999
    if rank <= 10:
        tone = "Strong top-tier match"
    elif rank <= 30:
        tone = "Solid fit"
    elif rank <= 60:
        tone = "Qualified but not ideal"
    else:
        tone = "Marginal fit"

    company_part = f" at {company}" if company else ""
    career_part = ""
    if recent_role and recent_company:
        career_part = f" Recent role: {recent_role} at {recent_company}."

    base = (
        f"{tone}: {title}{company_part} with {yoe} years experience.{career_part} "
        f"Matched JD skills: {skills_text}. Redrob signals: {signals_text}."
    )

    if concerns:
        return f"{base} Concerns: {'; '.join(concerns)}."
    return base


def rerank_top_candidates(top_candidates: list[dict], parsed_jd: dict, use_llm: bool = False) -> list[dict]:
    """
    Assign ranks and generate reasoning. Caps scores before sort so CSV tie-break
    matches model order (submission_spec: equal scores break by candidate_id ascending).
    """
    final_ranked = []

    llm_success = False
    if use_llm:
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                print("Using Gemini API for candidate reasoning...")
                for cand in top_candidates:
                    prompt = (
                        f"You are a professional tech recruiter. Given candidate profile summary: "
                        f"'{cand['profile']['summary']}', current title: '{cand['profile']['current_title']}', "
                        f"experience: {cand['profile']['years_of_experience']} years, "
                        f"skills: {[s['name'] for s in cand['skills']]}, and target job: "
                        f"'{parsed_jd['role_title']}'. Write one concise sentence (under 30 words) "
                        f"on fit for this role. Be specific; cite profile facts only."
                    )
                    response = model.generate_content(prompt)
                    cand["_reasoning"] = response.text.strip().replace('"', "")
                llm_success = True
            except Exception as e:
                print(f"Failed to use Gemini API: {e}. Falling back to rule-based reasoning.")

        elif openai_key and not llm_success:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                print("Using OpenAI API for candidate reasoning...")
                for cand in top_candidates:
                    prompt = (
                        f"Write one recruiter sentence (under 25 words) for: "
                        f"Title: {cand['profile']['current_title']}, YoE: "
                        f"{cand['profile']['years_of_experience']}, Skills: "
                        f"{[s['name'] for s in cand['skills']]}. Role: {parsed_jd['role_title']}. "
                        f"Cite profile facts only."
                    )
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                    )
                    cand["_reasoning"] = response.choices[0].message.content.strip().replace('"', "")
                llm_success = True
            except Exception as e:
                print(f"Failed to use OpenAI API: {e}. Falling back to rule-based reasoning.")

    for cand in top_candidates:
        cand["_final_score"] = min(99.4, max(0.0, cand["_final_score"]))

    sorted_candidates = sorted(
        top_candidates,
        key=lambda x: (-round(x["_final_score"] / 100.0, 4), x["candidate_id"]),
    )

    target_skills = set(parsed_jd.get("required_skills", []) + parsed_jd.get("nice_to_have_skills", []))

    if not llm_success:
        for rank, cand in enumerate(sorted_candidates, 1):
            cand["_rank"] = rank
            cand["_reasoning"] = generate_rule_based_reasoning(
                cand, cand["_final_score"], cand["_breakdown"], target_skills, rank
            )
    else:
        for rank, cand in enumerate(sorted_candidates, 1):
            cand["_rank"] = rank

    return sorted_candidates
