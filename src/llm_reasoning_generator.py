"""
Script to update candidate reasonings in outputs/mohd_ibadullah.csv.
Uses Gemini 2.5 Flash Lite with automatic failover to Groq (Llama 3.3 70B)
to handle the Gemini API key's daily quota limit.
"""
import os
import sys
import json
import time
import random
import requests
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import stream_candidates
from src.jd_parser import parse_job_description


def get_gemini_reasoning(api_key, model_name, candidate_id, rank, profile, parsed_jd, target_skills):
    """
    Queries Gemini API to generate reasoning.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    current_title = profile.get("current_title", "Software Engineer")
    years_of_experience = profile.get("years_of_experience", 0.0)
    
    skills_list = [s.get("name", "") for s in target_skills]
    skills_str = ", ".join(skills_list[:8])
    
    summary = profile.get("summary", "").replace("\n", " ").strip()
    if len(summary) > 300:
        summary = summary[:300] + "..."

    prompt = (
        f"Role being filled: '{parsed_jd['role_title']}'\n"
        f"Candidate ID: {candidate_id}\n"
        f"Candidate Title: {current_title}\n"
        f"Candidate YoE: {years_of_experience} years\n"
        f"Candidate Skills: {skills_str}\n"
        f"Candidate Summary: {summary}\n"
        f"System Rank: Ranked #{rank} out of 100,000 candidates.\n\n"
        f"Task:\n"
        f"Write exactly ONE natural sentence under 25 words explaining why this candidate is a strong fit. "
        f"MUST include their rank (e.g., 'Ranked #{rank}'), their job title '{current_title}', and highlight one or two key skills from the list: {skills_str}. "
        f"Reference candidate ID '{candidate_id}' instead of names. "
        f"Do NOT include double quotes or backslashes in the output. Avoid corporate fluff."
    )
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 60
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=12)
        if response.status_code == 200:
            res_data = response.json()
            candidates = res_data.get('candidates', [])
            if not candidates:
                return None, False
            
            candidate = candidates[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            if not parts:
                return None, False
                
            text = parts[0].get('text', '')
            clean_text = text.strip().replace('"', '').replace('\\', '').replace('\n', ' ')
            return clean_text, False
        
        elif response.status_code == 429:
            print(f"Gemini API daily/minute limit reached (429) for {candidate_id}.")
            return None, True  # True means Gemini is exhausted
        else:
            print(f"Gemini API returned error {response.status_code} for {candidate_id}")
            return None, False
            
    except Exception as e:
        print(f"Gemini API request exception for {candidate_id}: {e}")
        return None, False


def get_groq_reasoning(api_key, model_name, candidate_id, rank, profile, parsed_jd, target_skills):
    """
    Queries Groq API (Llama 3.3 70B) to generate reasoning as a failover.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    current_title = profile.get("current_title", "Software Engineer")
    years_of_experience = profile.get("years_of_experience", 0.0)
    
    skills_list = [s.get("name", "") for s in target_skills]
    skills_str = ", ".join(skills_list[:8])
    
    summary = profile.get("summary", "").replace("\n", " ").strip()
    if len(summary) > 300:
        summary = summary[:300] + "..."

    prompt = (
        f"Role being filled: '{parsed_jd['role_title']}'\n"
        f"Candidate ID: {candidate_id}\n"
        f"Candidate Title: {current_title}\n"
        f"Candidate YoE: {years_of_experience} years\n"
        f"Candidate Skills: {skills_str}\n"
        f"Candidate Summary: {summary}\n"
        f"System Rank: Ranked #{rank} out of 100,000 candidates.\n\n"
        f"Task:\n"
        f"Write exactly ONE natural sentence under 25 words explaining why this candidate is a strong fit. "
        f"MUST include their rank (e.g., 'Ranked #{rank}'), their job title '{current_title}', and highlight one or two key skills from the list: {skills_str}. "
        f"Reference candidate ID '{candidate_id}' instead of names. "
        f"Do NOT include double quotes or backslashes in the output. Avoid corporate fluff."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a professional tech recruiter writing concise candidate evaluations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 50
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=12)
        if response.status_code == 200:
            res_data = response.json()
            text = res_data['choices'][0]['message']['content']
            clean_text = text.strip().replace('"', '').replace('\\', '').replace('\n', ' ')
            return clean_text
        else:
            print(f"Groq API returned error {response.status_code} for {candidate_id}: {response.text}")
            return None
    except Exception as e:
        print(f"Groq API request exception for {candidate_id}: {e}")
        return None


def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Load .env variables
    load_dotenv(dotenv_path=project_root / ".env")

    # Try to find default candidates file path
    default_candidates_paths = [
        project_root.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
        project_root / "candidates.jsonl",
        Path("c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    ]
    
    default_candidates = None
    for p in default_candidates_paths:
        if p.exists():
            default_candidates = str(p)
            break

    parser = argparse.ArgumentParser(description="Update reasonings using Gemini/Groq")
    parser.add_argument("--candidates", default=default_candidates or "./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=str(project_root / "config" / "job_description.json"),
                        help="Path to job_description.json")
    parser.add_argument("--submission", default=str(project_root / "outputs" / "mohd_ibadullah.csv"),
                        help="Path to our submission CSV to update")
    parser.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Gemini API Key")
    parser.add_argument("--groq-key", default=os.environ.get("api_key") or os.environ.get("GROQ_API_KEY"),
                        help="Groq API Key")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite",
                        help="Gemini model name")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: Candidates file not found at {candidates_path.resolve()}")
        sys.exit(1)

    # Load Job Description config
    jd_config_path = Path(args.jd)
    if not jd_config_path.exists():
        print(f"Error: JD config file not found at {jd_config_path}")
        sys.exit(1)

    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)

    parsed_jd = parse_job_description(jd_input)

    # Load Submission CSV
    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: Submission CSV not found at {sub_path}")
        sys.exit(1)

    df_sub = pd.read_csv(sub_path)
    # Ensure sorted by rank
    df_sub = df_sub.sort_values("rank")
    candidates_to_process = df_sub.to_dict(orient="records")

    print(f"Loaded {len(candidates_to_process)} candidates from {sub_path.name} to process.")

    # Load all candidate profiles into a lookup dictionary
    print("Loading candidate profiles for LLM context...")
    cand_ids_to_find = set(c["candidate_id"] for c in candidates_to_process)
    cand_lookup = {}
    
    for cand in stream_candidates(str(candidates_path)):
        cid = cand["candidate_id"]
        if cid in cand_ids_to_find:
            cand_lookup[cid] = cand
            if len(cand_lookup) == len(cand_ids_to_find):
                break

    print(f"Loaded profiles for all {len(cand_lookup)} target candidates.")

    # Start generation loop
    print("Generating reasonings via Gemini with automatic failover to Groq...")
    updated_count = 0
    fallback_count = 0
    
    gemini_exhausted = False
    
    for index, item in enumerate(candidates_to_process):
        cid = item["candidate_id"]
        rank = item["rank"]
        
        # Get candidate profile details
        cand_data = cand_lookup.get(cid)
        if not cand_data:
            print(f"Warning: profile data not found for {cid}, keeping original reasoning.")
            fallback_count += 1
            continue
            
        profile = cand_data.get("profile", {})
        skills = cand_data.get("skills", [])
        
        reasoning_text = None
        
        if not gemini_exhausted and args.gemini_key:
            print(f"[{index+1}/{len(candidates_to_process)}] Querying Gemini for {cid} (Rank #{rank})...")
            reasoning_text, gemini_exhausted = get_gemini_reasoning(
                api_key=args.gemini_key,
                model_name=args.gemini_model,
                candidate_id=cid,
                rank=rank,
                profile=profile,
                parsed_jd=parsed_jd,
                target_skills=skills
            )
            if reasoning_text:
                item["reasoning"] = reasoning_text
                updated_count += 1
                time.sleep(4.5)  # 4.5s delay for Gemini
                continue
                
        if gemini_exhausted or not reasoning_text:
            if args.groq_key:
                print(f"[{index+1}/{len(candidates_to_process)}] Querying Groq (Failover) for {cid} (Rank #{rank})...")
                reasoning_text = get_groq_reasoning(
                    api_key=args.groq_key,
                    model_name=args.groq_model,
                    candidate_id=cid,
                    rank=rank,
                    profile=profile,
                    parsed_jd=parsed_jd,
                    target_skills=skills
                )
                if reasoning_text:
                    item["reasoning"] = reasoning_text
                    updated_count += 1
                    time.sleep(2.0)  # 2.0s delay for Groq
                    continue
            
        # Fallback to original
        print(f"Keeping original reasoning for {cid} (fallback).")
        fallback_count += 1
        time.sleep(0.5)

    print(f"\nCompleted: {updated_count} reasonings updated, {fallback_count} kept as original fallbacks.")

    # Write updated rows back
    df_out = pd.DataFrame(candidates_to_process)
    df_out = df_out[["candidate_id", "rank", "score", "reasoning"]]
    
    # Save output
    df_out.to_csv(sub_path, index=False)
    print(f"Updated CSV saved to {sub_path}")
    
    # Also save to the outer copy in the nesting folder if exists
    outer_path = Path("c:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/mohd_ibadullah.csv")
    if outer_path.exists() or outer_path.parent.exists():
        os.makedirs(outer_path.parent, exist_ok=True)
        df_out.to_csv(outer_path, index=False)
        print(f"Copy saved to {outer_path}")


if __name__ == "__main__":
    main()
