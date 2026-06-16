# Monkeypatch Starlette dependencies to support Streamlit's custom server integrations
try:
    import starlette.middleware.gzip
    if not hasattr(starlette.middleware.gzip, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
        starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = ("text/event-stream",)
        
    if not hasattr(starlette.middleware.gzip, "IdentityResponder"):
        class MockIdentityResponder:
            def __init__(self, app, minimum_size):
                self.app = app
                self.minimum_size = minimum_size
                self.send = None
            async def __call__(self, scope, receive, send):
                self.send = send
                await self.app(scope, receive, self.send_with_compression)
            async def send_with_compression(self, message):
                await self.send(message)
        starlette.middleware.gzip.IdentityResponder = MockIdentityResponder
except ImportError:
    pass

import streamlit as st
import pandas as pd
import json
import os
import sys
import time
import requests

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.jd_parser import parse_job_description
from src.bm25_filter import BM25Filter
from src.honeypot_detector import detect_trap
from src.embedding_scorer import EmbeddingScorer
from src.feature_scorer import calculate_candidate_score
from src.llm_reranker import rerank_top_candidates

def get_api_key():
    # 1. Check env var GEMINI_API_KEY
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # 2. Check env var api_key (from .env)
    key = os.environ.get("api_key")
    if key:
        return key
    # 3. Read .env file manually
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        if k.strip().lower() in ["gemini_api_key", "api_key"]:
                            return v.strip().strip('"').strip("'")
        except Exception:
            pass
    return None

def generate_interview_questions(jd_text, candidate):
    api_key = get_api_key()
    if not api_key:
        return None, "Add API Key (GEMINI_API_KEY or api_key) to .env to enable this feature"
        
    profile = candidate.get("profile", {})
    skills_list = [s.get("name") for s in candidate.get("skills", [])]
    career_history = []
    for role in candidate.get("career_history", []):
        career_history.append(f"{role.get('title')} at {role.get('company')} ({role.get('duration_months', 0)} months): {role.get('description', '')[:100]}")
        
    candidate_desc = (
        f"Title: {profile.get('current_title')}\n"
        f"Years of Experience: {profile.get('years_of_experience')}\n"
        f"Skills: {', '.join(skills_list)}\n"
        f"Career History: {'; '.join(career_history)}"
    )
    
    prompt = (
        f"Given this JD:\n{jd_text}\n\n"
        f"And candidate profile:\n{candidate_desc}\n\n"
        f"Generate exactly 3 targeted technical interview questions that probe gaps between the candidate and JD requirements. "
        f"Format as a numbered list. Keep each question under 2 lines."
    )
    
    if api_key.startswith("gsk_"):
        # Groq API compatibility
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                questions = res_json['choices'][0]['message']['content'].strip()
                return questions, None
            else:
                return None, f"Groq API Error (HTTP {response.status_code}): {response.text[:200]}"
        except Exception as e:
            return None, f"Connection error: {str(e)}"
    else:
        # Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                questions = res_json['candidates'][0]['content']['parts'][0]['text']
                return questions, None
            else:
                return None, f"Gemini API Error (HTTP {response.status_code}): {response.text[:200]}"
        except Exception as e:
            return None, f"Connection error: {str(e)}"


if __name__ == "__main__":
    # Set page configuration with premium dark aesthetic look
    st.set_page_config(
        page_title="TalentLens AI — Candidate Discovery & Ranking",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom premium styling
    st.markdown("""
    <style>
        /* Dark Theme Background & Fonts */
        .stApp {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            font-family: 'Outfit', 'Inter', sans-serif;
            font-weight: 700;
            color: #ffffff;
        }
        .main-title {
            background: linear-gradient(90deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #8892b0;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        /* Metric Cards */
        .metric-card {
            background-color: #1b1f2b;
            border: 1px solid #2d313f;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 700;
            color: #06b6d4;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8892b0;
        }
        /* Trap Flags */
        .decoy-alert {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid rgb(239, 68, 68);
            color: #fc8181;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
            margin: 5px 0;
        }
        .genuine-badge {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgb(16, 185, 129);
            color: #34d399;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    # Data paths — resolved relative to project root for portability (local + Streamlit Cloud)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SAMPLE_PATH = os.path.join(_PROJECT_ROOT, "data", "sample_candidates.json")
    FULL_PATH = os.path.join(_PROJECT_ROOT, "data", "candidates.jsonl")  # Only available locally
    DEFAULT_JD_PATH = os.path.join(_PROJECT_ROOT, "config", "job_description.json")
    _FULL_DATASET_AVAILABLE = os.path.exists(FULL_PATH)

    @st.cache_resource
    def load_embedding_model():
        """Cache the embedding model locally so it's loaded only once."""
        return EmbeddingScorer()

    @st.cache_data
    def load_datasets():
        """Load both datasets and keep them cached."""
        # 1. Load Sample Dataset
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            sample_candidates = json.load(f)

        # 2. Load lightweight version of Full Dataset for BM25
        full_lightweight = []
        if os.path.exists(FULL_PATH):
            with open(FULL_PATH, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    cand = json.loads(line)
                    profile = cand.get("profile", {})
                    skills = [s.get("name", "") for s in cand.get("skills", [])]

                    # Store seek position or just stream line-by-line using index
                    lightweight_cand = {
                        "candidate_id": cand.get("candidate_id"),
                        "profile": {
                            "anonymized_name": profile.get("anonymized_name", "Anonymous"),
                            "current_title": profile.get("current_title", ""),
                            "current_company": profile.get("current_company", ""),
                            "current_company_size": profile.get("current_company_size", ""),
                            "current_industry": profile.get("current_industry", ""),
                            "headline": profile.get("headline", ""),
                            "summary": profile.get("summary", ""),
                            "years_of_experience": float(profile.get("years_of_experience", 0.0)),
                            "location": profile.get("location", ""),
                            "country": profile.get("country", "")
                        },
                        "skills": cand.get("skills", []),
                        "career_history": cand.get("career_history", []),
                        "education": cand.get("education", []),
                        "certifications": cand.get("certifications", []),
                        "redrob_signals": cand.get("redrob_signals", {}),
                        "_line_index": idx
                    }
                    full_lightweight.append(lightweight_cand)

        return sample_candidates, full_lightweight

    # App layout
    st.markdown('<div class="main-title">TalentLens AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Candidate Discovery, Semantic Matching & Honeypot Detection</div>', unsafe_allow_html=True)

    # Load resources
    with st.spinner("Initializing models and indexing candidates..."):
        embed_scorer = load_embedding_model()
        sample_cands, full_cands = load_datasets()

    # Default JD loading
    default_jd_text = ""
    if os.path.exists(DEFAULT_JD_PATH):
        with open(DEFAULT_JD_PATH, "r", encoding="utf-8") as f:
            jd_dict = json.load(f)
            default_jd_text = (
                f"Role: {jd_dict.get('role_title', '')}\n"
                f"Must-have skills: {', '.join(jd_dict.get('required_skills', []))}\n"
                f"Nice-to-have: {', '.join(jd_dict.get('nice_to_have_skills', []))}\n"
                f"Min experience: {jd_dict.get('min_years_experience', '')}+ years\n"
                f"Seniority: {jd_dict.get('seniority_level', '')}\n"
                f"Domain: {', '.join(jd_dict.get('domain_keywords', []))}"
            )

    # Sidebar
    st.sidebar.header("🔧 Configuration")

    if _FULL_DATASET_AVAILABLE:
        dataset_type = st.sidebar.selectbox(
            "Select Dataset Source",
            options=["Sample Dataset (50 candidates)", "Full Dataset (100,000 candidates)"],
            index=1  # Default to Full Dataset for demo impact
        )
    else:
        dataset_type = "Sample Dataset (50 candidates)"
        st.sidebar.info("🔍 Running on Sample Dataset (50 candidates). Full dataset available in local mode only.")

    # Custom Weights tuning
    st.sidebar.subheader("⚖️ Scoring Formula Weights")
    w_sim = st.sidebar.slider("Semantic Similarity Weight", 0.0, 1.0, 0.35, 0.05)
    w_skill = st.sidebar.slider("Skills Overlap Weight", 0.0, 1.0, 0.30, 0.05)
    w_title = st.sidebar.slider("Title/YoE Match Weight", 0.0, 1.0, 0.15, 0.05)
    w_signals = st.sidebar.slider("Engagement Signals Weight", 0.0, 1.0, 0.10, 0.05)
    w_trap = st.sidebar.slider("Honeypot Penalty Weight", 0.0, 1.0, 0.40, 0.05)

    weights = {
        "semantic_similarity": w_sim,
        "skill_match_score": w_skill,
        "title_seniority_match": w_title,
        "signal_bonus": w_signals,
        "trap_penalty": w_trap
    }

    max_results = st.sidebar.number_input("Maximum Results to Rank", min_value=5, max_value=200, value=20)

    # Layout: Main columns
    col_jd, col_results = st.columns([1, 2.2])

    with col_jd:
        st.subheader("📝 Target Job Description")
        jd_input = st.text_area(
            "Paste Job Description here:",
            value=default_jd_text,
            height=400
        )

        run_btn = st.button("🔍 Discover & Rank Candidates", use_container_width=True, type="primary")

    # Execute Search
    if run_btn or 'ranked_results' not in st.session_state:
        st.session_state['run_pipeline'] = True

    if st.session_state.get('run_pipeline', False):
        st.session_state['run_pipeline'] = False

        t_start = time.time()

        # 1. Parse JD
        parsed_jd = parse_job_description(jd_input)

        # Select working dataset
        working_cands = sample_cands if dataset_type.startswith("Sample") else full_cands

        with st.spinner("Filtering candidates using BM25..."):
            # 2. Stage 1 BM25 Lexical Filter
            bm25_filter = BM25Filter(working_cands)
            # Limit lexical search: for Streamlit responsiveness, score top 500 candidates
            top_filter_limit = 500 if dataset_type.startswith("Full") else 50
            filtered_candidates = bm25_filter.filter_candidates(parsed_jd, top_n=top_filter_limit)

        with st.spinner("Computing semantic embeddings & decoy scores..."):
            # 3. Deep Feature Scoring on filtered subset
            candidate_texts = []
            for cand in filtered_candidates:
                profile = cand.get("profile", {})
                title = profile.get("current_title", "")
                headline = profile.get("headline", "")
                summary = profile.get("summary", "")
                skills_str = ", ".join([s.get("name", "") for s in cand.get("skills", [])[:15]])
                career_titles = " | ".join([r.get("title", "") for r in cand.get("career_history", [])[:5]])
                candidate_texts.append(f"{title}. {headline}. {summary} Skills: {skills_str}. Career: {career_titles}")

            jd_embedding_text = (
                f"{parsed_jd['role_title']}. "
                f"Required skills: {', '.join(parsed_jd['required_skills'])}. "
                f"Nice to have: {', '.join(parsed_jd['nice_to_have_skills'])}. "
                f"Domain: {', '.join(parsed_jd.get('domain_keywords', []))}. "
                f"Seniority: {parsed_jd.get('seniority_level', 'senior')}"
            )
            similarities = embed_scorer.compute_similarity(jd_embedding_text, candidate_texts)

            scored_candidates = []
            for cand, sim in zip(filtered_candidates, similarities):
                trap_score, trap_reason = detect_trap(cand)
                cand["_trap_score"] = trap_score
                cand["_trap_reason"] = trap_reason

                final_score, breakdown = calculate_candidate_score(cand, sim, trap_score, parsed_jd, weights=weights)
                cand["_final_score"] = final_score
                cand["_breakdown"] = breakdown
                scored_candidates.append(cand)

            # 4. Reranking and reasoning
            ranked = rerank_top_candidates(scored_candidates, parsed_jd, use_llm=False)
            st.session_state['ranked_results'] = ranked
            st.session_state['duration'] = time.time() - t_start
            st.session_state['total_candidates'] = len(working_cands)
            st.session_state['parsed_jd'] = parsed_jd

    # Render Results
    if 'ranked_results' in st.session_state:
        ranked = st.session_state['ranked_results']
        duration = st.session_state['duration']
        total_candidates = st.session_state['total_candidates']
        parsed_jd = st.session_state['parsed_jd']

        with col_results:
            # Display Stats
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{total_candidates:,}</div><div class="metric-label">Candidates Scanned</div></div>', unsafe_allow_html=True)
            with col_s2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(ranked)}</div><div class="metric-label">Candidates Scored</div></div>', unsafe_allow_html=True)
            with col_s3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{duration:.2f}s</div><div class="metric-label">Search Duration</div></div>', unsafe_allow_html=True)

            st.subheader("🏆 Discovery Ranking (Top Matches)")

            # Generate downloadable CSV from ranked results
            csv_rows = []
            for idx, cand in enumerate(ranked[:100]):
                csv_rows.append({
                    "candidate_id": cand["candidate_id"],
                    "rank": cand["_rank"],
                    "score": round(cand["_final_score"] / 100.0, 4),
                    "reasoning": cand["_reasoning"]
                })
            df_download = pd.DataFrame(csv_rows)
            csv_data = df_download.to_csv(index=False)
            st.download_button(
                label="📥 Download Ranked CSV",
                data=csv_data,
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Display candidates
            for idx in range(min(max_results, len(ranked))):
                cand = ranked[idx]
                profile = cand.get("profile", {})
                breakdown = cand["_breakdown"]

                trap_score = cand["_trap_score"]
                is_trap = trap_score > 0.4

                with st.container():
                    # Card outline
                    card_style = "border: 1px solid #ef4444; background-color: #211515;" if is_trap else "border: 1px solid #2d313f; background-color: #1b1f2b;"

                    st.markdown(f"""
                    <div style="border-radius: 8px; padding: 15px; margin-bottom: 12px; {card_style}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.25rem; font-weight: bold; color: #ffffff;">#{cand['_rank']} — {profile.get('anonymized_name', 'Anonymous')}</span>
                                <span style="font-size: 0.85rem; color: #8892b0; margin-left: 10px;">({cand['candidate_id']})</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: {'#ef4444' if is_trap else '#34d399'};">
                                {cand['_final_score']:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Columns inside candidate card
                    c_info, c_scores = st.columns([2, 1])

                    with c_info:
                        company = profile.get('current_company', '')
                        industry = profile.get('current_industry', '')
                        company_display = f"**{company}**" if company else "N/A"
                        industry_display = f" ({industry})" if industry else ""
                        st.markdown(f"**Current Title:** `{profile.get('current_title', 'N/A')}` at {company_display}{industry_display} — {profile.get('years_of_experience', 0)} YoE")
                        st.markdown(f"📍 {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
                        st.markdown(f"**Summary:** *\"{profile.get('summary', '')[:200]}...\"*")

                        # Honeypot warning
                        if is_trap:
                            st.markdown(f'<div class="decoy-alert">⚠️ <b>Honeypot Trap Detected:</b> {cand["_trap_reason"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="genuine-badge">✓ Verified Match Profile</span>', unsafe_allow_html=True)

                        # Reasoning
                        st.markdown(f"**System Recruiter Rationale:**\n*{cand['_reasoning']}*")

                    with c_scores:
                        st.write("**Score Breakdown:**")
                        st.write(f"- Semantic Similarity: `{breakdown['semantic_similarity']*100:.1f}%`")
                        st.write(f"- Skills Matching Fit: `{breakdown['skill_match_score']*100:.1f}%`")
                        st.write(f"- Title/Experience Fit: `{breakdown['title_seniority_match']*100:.1f}%`")
                        st.write(f"- Activity & Signals: `{breakdown['signal_bonus']*100:.1f}%`")
                        if breakdown['trap_penalty_applied'] > 0:
                            st.write(f"- Decoy Mismatch Penalty: `-{breakdown['trap_penalty_applied']:.1f} pts`")

                    # Details expander
                    with st.expander("🔍 View Full Profile & Career History"):
                        # Skills
                        st.markdown("**Skills:**")
                        skills_list = [f"{s.get('name')} ({s.get('proficiency')})" for s in cand.get("skills", [])]
                        st.write(", ".join(skills_list))

                        # Career history
                        st.markdown("**Career History:**")
                        for role in cand.get("career_history", []):
                            st.markdown(f"- **{role.get('title')}** at *{role.get('company')}* ({role.get('duration_months', 0)} months)")
                            st.markdown(f"  *Description:* {role.get('description')}")

                        # Signals
                        st.markdown("**Redrob Signals:**")
                        signals = cand.get("redrob_signals", {})
                        st.json({k: v for k, v in signals.items() if v != -1})

                        # Education
                        education = cand.get("education", [])
                        if education:
                            st.markdown("**Education:**")
                            for edu in education:
                                degree = edu.get('degree', 'Degree')
                                field = edu.get('field_of_study', '')
                                institution = edu.get('institution', 'Unknown')
                                years = f"{edu.get('start_year', '')}-{edu.get('end_year', '')}"
                                tier = edu.get('tier', '')
                                tier_badge = f" `{tier}`" if tier and tier != 'unknown' else ""
                                st.markdown(f"- **{degree}** in *{field}* from **{institution}** ({years}){tier_badge}")

                        # Suggested Interview Questions Section
                        if not is_trap:
                            st.markdown("<hr style='border: 0.5px solid #2d313f; margin: 10px 0;'>", unsafe_allow_html=True)
                            st.markdown("**🎯 Suggested Interview Questions**")

                            q_key = f"questions_{cand['candidate_id']}"
                            if q_key not in st.session_state:
                                st.session_state[q_key] = None
                                st.session_state[f"{q_key}_err"] = None

                            if st.button("💬 Generate Targeted Questions", key=f"btn_{cand['candidate_id']}"):
                                with st.spinner("Analyzing profile gaps and generating questions..."):
                                    questions, err = generate_interview_questions(jd_input, cand)
                                    st.session_state[q_key] = questions
                                    st.session_state[f"{q_key}_err"] = err

                            if st.session_state[q_key]:
                                st.info(st.session_state[q_key])
                                st.caption("⚠️ AI-generated questions — verify before use")
                            elif st.session_state[f"{q_key}_err"]:
                                st.error(st.session_state[f"{q_key}_err"])

                    st.markdown("<hr style='border: 0.5px solid #2d313f; margin: 10px 0;'>", unsafe_allow_html=True)
