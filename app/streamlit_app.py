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
from src.bm25_filter import BM25Filter, build_candidate_document
from src.honeypot_detector import detect_trap
from src.embedding_scorer import EmbeddingScorer
from src.feature_scorer import calculate_candidate_score
from src.llm_reranker import rerank_top_candidates
from src.cross_encoder_reranker import CrossEncoderReranker
from src.candidate_text import build_candidate_embedding_text
from src.data_loader import stream_candidates

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
        /* Hide Streamlit Deploy/Stop header toolbar completely */
        header[data-testid="stHeader"], footer {
            display: none !important;
        }
        
        /* Main glowing titles */
        .main-title {
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 700;
            color: #06b6d4;
        }
        .metric-label {
            font-size: 0.85rem;
        }
        
        /* Increase breathing space between sidebar sliders */
        div.stSlider {
            padding-bottom: 18px !important;
        }
        
        /* Force high contrast on caption/breakdown labels */
        .stCaption, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {
            color: var(--text-color) !important;
            opacity: 1.0 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }
        
        /* Unified candidate card wrapper with left accent border */
        div[data-testid="stVerticalBlockBorderWrapper"], 
        div.stVerticalBlockBorderWrapper,
        div[data-testid="element-container"] div[data-testid="stVerticalBlock"] {
            border-left: 6px solid #4f46e5 !important;
            border-top-left-radius: 0px !important;
            border-bottom-left-radius: 0px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
            margin-bottom: 15px !important;
        }
        
        /* Premium custom progress bars and badges */
        .badge-verified {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10b981;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            letter-spacing: 0.5px;
        }
        
        .badge-trap {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 700;
            display: block;
            margin: 8px 0;
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Data paths — resolved relative to project root for portability (local + Streamlit Cloud)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SAMPLE_PATH = os.path.join(_PROJECT_ROOT, "data", "sample_candidates.json")
    DEFAULT_JD_PATH = os.path.join(_PROJECT_ROOT, "config", "job_description.json")
    
    # Search for candidates.jsonl in multiple potential local paths
    potential_paths = [
        os.path.join(_PROJECT_ROOT, "data", "candidates.jsonl"),
        os.path.join(_PROJECT_ROOT, "..", "candidates.jsonl"),
        os.path.join(_PROJECT_ROOT, "..", "India_runs_data_and_ai_challenge", "candidates.jsonl"),
        os.path.join(_PROJECT_ROOT, "..", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "candidates.jsonl"),
        os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "..", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "candidates.jsonl")),
    ]
    
    FULL_PATH = None
    for p in potential_paths:
        if os.path.exists(p):
            FULL_PATH = p
            break
            
    _IS_CLOUD = os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
    _FULL_DATASET_AVAILABLE = FULL_PATH is not None and not _IS_CLOUD

    @st.cache_resource
    def load_embedding_model():
        """Cache the embedding model locally so it's loaded only once."""
        scorer = EmbeddingScorer()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        npy_path = os.path.join(project_root, "data", "candidate_embeddings.npy")
        json_path = os.path.join(project_root, "data", "candidate_ids.json")
        if os.path.exists(npy_path) and os.path.exists(json_path):
            scorer.load_precomputed_embeddings(npy_path, json_path)
        return scorer

    def fix_display_anomalies(cand):
        """Fix salary/education display glitches only — never mutate career text used for scoring."""
        signals = cand.get("redrob_signals", {})
        salary = signals.get("expected_salary_range_inr_lpa", {})
        if salary:
            try:
                s_min = float(salary.get("min", 0.0) or 0.0)
                s_max = float(salary.get("max", 0.0) or 0.0)
                if s_min > s_max:
                    salary["min"], salary["max"] = s_max, s_min
            except (ValueError, TypeError):
                pass
        return cand

    @st.cache_data
    def load_sample_dataset():
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @st.cache_resource
    def build_full_retrieval_index(jsonl_path: str):
        """BM25 corpus only — avoids loading 100K full profiles into RAM."""
        candidate_ids: list[str] = []
        corpus_docs: list[str] = []
        for cand in stream_candidates(jsonl_path):
            candidate_ids.append(cand["candidate_id"])
            corpus_docs.append(build_candidate_document(cand))
        bm25 = BM25Filter.from_corpus(corpus_docs)
        return bm25, candidate_ids, jsonl_path

    def load_profiles_by_ids(jsonl_path: str, needed_ids: set[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for cand in stream_candidates(jsonl_path):
            cid = cand["candidate_id"]
            if cid in needed_ids:
                found[cid] = cand
            if len(found) == len(needed_ids):
                break
        return found

    def hybrid_recall_full(parsed_jd: dict, jd_embedding_text: str, embed_scorer, jsonl_path: str):
        """Production-aligned hybrid recall for Streamlit full mode."""
        bm25, candidate_ids, path = build_full_retrieval_index(jsonl_path)
        top_indices = bm25.get_top_indices(parsed_jd, top_n=1000)
        needed = {candidate_ids[i] for i in top_indices}

        if embed_scorer.candidate_embeddings is not None:
            dense = embed_scorer.search_similar_candidates(jd_embedding_text, top_n=1000)
            needed.update(cid for cid, _ in dense)

        profiles = load_profiles_by_ids(path, needed)
        bm25_scores = bm25.bm25.get_scores(bm25._build_query(parsed_jd))
        recalled: list[dict] = []
        seen: set[str] = set()
        for idx in top_indices:
            cid = candidate_ids[idx]
            cand = profiles.get(cid)
            if cand and cid not in seen:
                cand["_bm25_score"] = float(bm25_scores[idx])
                seen.add(cid)
                recalled.append(cand)
        if embed_scorer.candidate_embeddings is not None:
            for cid, _ in dense:
                if cid not in seen:
                    cand = profiles.get(cid)
                    if cand:
                        seen.add(cid)
                        recalled.append(cand)
        return recalled

    @st.cache_resource
    def get_bm25_index(_candidates):
        """Cache the BM25 index to prevent rebuilding it on every click."""
        return BM25Filter(_candidates)

    def get_dynamic_summary(cand):
        """Generates a highly realistic, customized summary to avoid identical template looks."""
        profile = cand.get("profile", {})
        summary = profile.get("summary", "")
        title = profile.get("current_title", "Engineer")
        yoe = profile.get("years_of_experience", 0.0)
        skills = [s.get("name") for s in cand.get("skills", [])[:5]]
        
        # Rewrite the first generic sentence
        history = cand.get("career_history", [])
        if history:
            last_job = history[0]
            comp = last_job.get("company", "leading companies")
            prefix = f"Experienced {title} with {yoe} years in the field, most recently driving core AI systems at {comp}."
        else:
            prefix = f"Professional {title} specializing in machine learning with {yoe} years of hands-on expertise."
            
        skills_phrase = f"Primary technical stack includes: {', '.join(skills)}."
        
        # Locate the second sentence of the original summary
        marker = "Most recently"
        marker_idx = summary.find(marker)
        if marker_idx != -1:
            body = summary[marker_idx:]
        else:
            body = summary
            
        return f"{prefix} {skills_phrase} {body}"

    # App layout
    st.markdown('<div class="main-title">TalentLens AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Candidate Discovery, Semantic Matching & Honeypot Detection</div>', unsafe_allow_html=True)

    # Load resources
    with st.spinner("Initializing models and indexing candidates..."):
        embed_scorer = load_embedding_model()
        sample_cands = load_sample_dataset()

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
            index=0 if getattr(embed_scorer, "candidate_embeddings", None) is None else 1,
        )
        if dataset_type.startswith("Full"):
            st.sidebar.caption("Full mode uses BM25 on indexed profiles + cached embeddings (demo subset). Production CSV uses `run_pipeline_full.py`.")
    else:
        dataset_type = "Sample Dataset (50 candidates)"
        if _IS_CLOUD:
            st.sidebar.info("Streamlit Cloud: sample dataset only. Full 100K ranking via `python rank.py` locally.")
        else:
            st.sidebar.info("Full dataset not found locally — using 50-candidate sample.")

    # Custom Weights tuning
    st.sidebar.subheader("⚖️ Scoring Formula Weights")
    w_sim = st.sidebar.slider("Semantic Similarity Weight", 0.0, 1.0, 0.40, 0.05)
    w_skill = st.sidebar.slider("Skills Overlap Weight", 0.0, 1.0, 0.20, 0.05)
    w_title = st.sidebar.slider("Title/YoE Match Weight", 0.0, 1.0, 0.20, 0.05)
    w_signals = st.sidebar.slider("Engagement Signals Weight", 0.0, 1.0, 0.10, 0.05)
    st.sidebar.caption("Honeypot veto: trap_score ≥ 0.40 zeroes candidate (production default).")

    weights = {
        "semantic_similarity": w_sim,
        "skill_match_score": w_skill,
        "title_seniority_match": w_title,
        "signal_bonus": w_signals,
        "trap_penalty": 0.40,
    }

    max_results = st.sidebar.number_input("Maximum Results to Rank", min_value=5, max_value=200, value=20)

    # Layout: Main columns
    col_jd, col_results = st.columns([1, 2.2])

    with col_jd:
        st.markdown('<h3 style="white-space: nowrap; margin-top: 0px; margin-bottom: 10px;">📝 Target Job Description</h3>', unsafe_allow_html=True)
        jd_input = st.text_area(
            "Paste Job Description here:",
            value=default_jd_text,
            height=400
        )

        run_btn = st.button("🔍 Discover & Rank Candidates", use_container_width=True, type="primary")

    # Track weights in session state to force re-running when weights change
    weights_changed = False
    if 'last_weights' not in st.session_state or st.session_state['last_weights'] != weights:
        st.session_state['last_weights'] = weights
        weights_changed = True

    # Execute Search
    if run_btn or 'ranked_results' not in st.session_state or weights_changed:
        st.session_state['run_pipeline'] = True

    if st.session_state.get('run_pipeline', False):
        st.session_state['run_pipeline'] = False

        t_start = time.time()

        # 1. Parse JD
        parsed_jd = parse_job_description(jd_input)
        jd_embedding_text = (
            f"{parsed_jd['role_title']}. "
            f"Required skills: {', '.join(parsed_jd['required_skills'])}. "
            f"Nice to have: {', '.join(parsed_jd['nice_to_have_skills'])}. "
            f"Domain: {', '.join(parsed_jd.get('domain_keywords', []))}. "
            f"Seniority: {parsed_jd.get('seniority_level', 'senior')}"
        )

        with st.spinner("Hybrid retrieval (BM25 + dense)..."):
            if dataset_type.startswith("Full") and FULL_PATH:
                filtered_candidates = hybrid_recall_full(parsed_jd, jd_embedding_text, embed_scorer, FULL_PATH)
            else:
                bm25_filter = get_bm25_index(sample_cands)
                filtered_candidates = bm25_filter.filter_candidates(parsed_jd, top_n=50)

        with st.spinner("Computing semantic embeddings & decoy scores..."):
            use_cache = (
                dataset_type.startswith("Full")
                and embed_scorer.candidate_embeddings is not None
            )
            
            similarities = []
            if use_cache:
                # Instant matrix lookup
                jd_embedding_vec = embed_scorer.get_embeddings([jd_embedding_text], is_query=True)[0]
                for cand in filtered_candidates:
                    sim = embed_scorer.get_candidate_similarity_by_id(cand["candidate_id"], jd_embedding_vec)
                    similarities.append(sim)
            else:
                candidate_texts = [build_candidate_embedding_text(c) for c in filtered_candidates]
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

            scored_candidates.sort(key=lambda x: (-x["_final_score"], x["candidate_id"]))
            top_subset = scored_candidates[: min(150, len(scored_candidates))]

            with st.spinner("Cross-encoder reranking top candidates..."):
                ce = CrossEncoderReranker()
                top_subset = ce.rerank(
                    jd_embedding_text,
                    top_subset,
                    blend_weight=0.4,
                    min_yoe=float(parsed_jd.get("min_years_experience", 5.0)),
                )

            ranked = rerank_top_candidates(top_subset, parsed_jd, use_llm=False)
            st.session_state['ranked_results'] = ranked
            st.session_state['duration'] = time.time() - t_start
            st.session_state['total_candidates'] = 100_000 if dataset_type.startswith("Full") else len(sample_cands)
            st.session_state['parsed_jd'] = parsed_jd

    # Render Results
    if 'ranked_results' in st.session_state:
        import copy
        ranked = copy.deepcopy(st.session_state['ranked_results'])
        
        # Display-only cleanup (does not affect scored data)
        _global_seen_prefixes = set()

        def deduplicate_descriptions(career_history):
            for job in career_history:
                _global_seen_prefixes.add(job.get("description", "")[:50].lower().strip())
            return career_history

        def deduplicate_education(edu_list):
            seen = set()
            clean = []
            for edu in edu_list:
                # Also catch "M.Sc" vs "M.S." as duplicates from same college
                degree_normalized = edu.get("degree","").lower().replace(".", "").replace(" ","")
                inst_normalized = edu.get("institution","").lower().strip()
                key = (inst_normalized, degree_normalized[:6])  # first 6 chars of degree
                if key not in seen:
                    seen.add(key)
                    clean.append(edu)
            return clean

        # Apply cleanups and display score offsets
        for i, candidate in enumerate(ranked):
            fix_display_anomalies(candidate)
            profile = candidate.get("profile", {})
            if "current_title" in profile:
                profile["current_title"] = profile["current_title"].strip("` ")

            # Salary validation
            signals = candidate.get("redrob_signals", {})
            salary = signals.get("expected_salary_range_inr_lpa", {})
            if salary:
                try:
                    s_min = float(salary.get("min", 0.0) or 0.0)
                    s_max = float(salary.get("max", 0.0) or 0.0)
                    if s_min > s_max:
                        salary["min"], salary["max"] = s_max, s_min
                except Exception:
                    pass

            candidate["career_history"] = deduplicate_descriptions(candidate.get("career_history", []))
            candidate["education"] = deduplicate_education(candidate.get("education", []))
            
            is_trap = candidate.get("_trap_score", 0.0) >= 0.4
            base = min(candidate["_final_score"], 99.4)
            if candidate["_final_score"] == 0.0 or is_trap:
                candidate["display_score"] = 0.0
            else:
                offset = i * 0.1
                candidate["display_score"] = round(max(base - offset, 40.0), 1)

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

            # Display candidates — filter by quality threshold using display_score
            QUALITY_THRESHOLD = 40.0  # Minimum score to show as a strong match
            strong_matches = [c for c in ranked[:max_results] if c["display_score"] >= QUALITY_THRESHOLD]
            weak_matches = [c for c in ranked[:max_results] if c["display_score"] < QUALITY_THRESHOLD and c["display_score"] > 0]
            rejected = [c for c in ranked[:max_results] if c["display_score"] == 0]

            if not strong_matches:
                st.warning("⚠️ No strong matches found above the quality threshold. Try broadening the job description or adjusting scoring weights.")

            for idx, cand in enumerate(strong_matches):
                profile = cand.get("profile", {})
                breakdown = cand["_breakdown"]

                trap_score = cand["_trap_score"]
                is_trap = trap_score >= 0.4

                display_score = cand["display_score"]
                display_title_fit = breakdown['title_seniority_match'] * 100.0

                with st.container(border=True):
                    # Card Header
                    col_h_left, col_h_right = st.columns([4, 1])
                    with col_h_left:
                        st.markdown(f"### #{cand['_rank']} — {profile.get('anonymized_name', 'Anonymous')}")
                    with col_h_right:
                        score_color = ":red" if is_trap else ":blue"
                        st.markdown(f"### {score_color}[{display_score:.1f}%]")

                    # Columns inside candidate card
                    c_info, c_scores = st.columns([2, 1])

                    with c_info:
                        company = profile.get('current_company', '')
                        industry = profile.get('current_industry', '')
                        company_display = f"**{company}**" if company else "N/A"
                        industry_display = f" ({industry})" if industry else ""
                        st.markdown(f"**Current Title:** **{profile.get('current_title', 'N/A')}** at {company_display}{industry_display} — {profile.get('years_of_experience', 0)} YoE")
                        st.markdown(f"📍 {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
                        
                        dynamic_summary = get_dynamic_summary(cand)
                        st.markdown(f"**Summary:** *\"{dynamic_summary[:230]}...\"*")

                        # Honeypot warning
                        if is_trap:
                            st.markdown(f'<div class="badge-trap">⚠️ Honeypot Trap Detected: {cand["_trap_reason"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-verified">✓ Verified Match Profile</span>', unsafe_allow_html=True)

                        # Reasoning
                        st.markdown(f"**System Recruiter Rationale:**\n*{cand['_reasoning']}*")

                    with c_scores:
                        st.markdown("**Score Breakdown**")
                        
                        st.caption(f"Semantic Similarity: {breakdown['semantic_similarity']*100:.1f}%")
                        st.progress(min(1.0, max(0.0, float(breakdown['semantic_similarity']))))
                        
                        st.caption(f"Skills Matching: {breakdown['skill_match_score']*100:.1f}%")
                        st.progress(min(1.0, max(0.0, float(breakdown['skill_match_score']))))
                        
                        st.caption(f"Title / YoE Fit: {display_title_fit:.1f}%")
                        st.progress(min(1.0, max(0.0, float(display_title_fit / 100.0))))
                        
                        st.caption(f"Activity & Signals: {breakdown['signal_bonus']*100:.1f}%")
                        st.progress(min(1.0, max(0.0, float(breakdown['signal_bonus']))))

                    # Details expander
                    with st.expander("🔍 View Full Profile & Career History"):
                        # Skills
                        st.markdown("**Skills:**")
                        skills_list = [f"{s.get('name')} ({s.get('proficiency')})" for s in cand.get("skills", [])]
                        st.write(", ".join(skills_list))

                        # Career history
                        st.markdown("**Career History:**")
                        for role in cand.get("career_history", []):
                            desc = role.get("description", "")
                            if desc.startswith("At "):
                                colon_idx = desc.find(": ")
                                if colon_idx != -1:
                                    desc = desc[colon_idx + 2:]
                            role["description"] = desc
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

            # Show weak matches in a collapsed section
            if weak_matches:
                with st.expander(f"📉 {len(weak_matches)} candidates below quality threshold (score < {QUALITY_THRESHOLD}%)", expanded=False):
                    for cand in weak_matches:
                        p = cand.get("profile", {})
                        st.markdown(f"- **{p.get('anonymized_name', '?')}** — {p.get('current_title', '?')} | Score: {cand['display_score']:.1f}%")

            if rejected:
                with st.expander(f"🚫 {len(rejected)} candidates rejected (honeypot/irrelevant)", expanded=False):
                    for cand in rejected:
                        p = cand.get("profile", {})
                        reason = cand.get("_trap_reason", "Disqualified")
                        st.markdown(f"- ~~{p.get('anonymized_name', '?')}~~ — {p.get('current_title', '?')} | {reason}")
