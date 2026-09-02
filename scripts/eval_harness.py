"""
Eval Harness — UMBRELA LLM-judge across multiple models.
Supports: Zen free models (OpenAI-compatible) + Gemini API.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import stream_candidates
from src.jd_parser import parse_job_description
from src.bm25_filter import BM25Filter


# ─── UMBRELA prompt (fixed, versioned) ──────────────────────────────────────
UMBRELA_PROMPT = """You are a relevance judge for a resume-ranking task.
Given a Job Description and a candidate profile, decide if the candidate is
"relevant" (would be in the top-10 for this role) or "not-relevant".

Output EXACTLY one line:
1 if relevant
0 if not-relevant

Do NOT include any reasoning, explanation, or extra text.

--- Job Description ---
{jd_text}

--- Candidate Profile ---
{candidate_text}

--- Decision:""".strip()


# ─── Model configs ───────────────────────────────────────────────────────────
ZEN_KEY = "REDACTED_API_KEY"
GEMINI_KEY = "REDACTED_API_KEY"

MODELS = {
    "nemotron-3-ultra-free": {
        "provider": "zen",
        "endpoint": "https://opencode.ai/zen/v1/chat/completions",
        "api_key": ZEN_KEY,
    },
    "nemotron-3.5-lightning-free": {
        "provider": "zen",
        "endpoint": "https://opencode.ai/zen/v1/chat/completions",
        "api_key": ZEN_KEY,
    },
    "laguna-s-2.1-free": {
        "provider": "zen",
        "endpoint": "https://opencode.ai/zen/v1/chat/completions",
        "api_key": ZEN_KEY,
    },
}


# ─── Query helpers ───────────────────────────────────────────────────────────
def query_zen(endpoint, api_key, prompt, model, temperature=0.0, max_tokens=512):
    """OpenAI-compatible chat completion (Zen free models)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(4):
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        if resp.status_code == 429:
            wait = min(30, 5 * (2 ** attempt))
            print(f"    429 rate limited, waiting {wait}s (attempt {attempt+1})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"].get("message", "Unknown error"))
        choices = data.get("choices", [])
        if not choices:
            raise Exception("No choices in response")
        content = choices[0].get("message", {}).get("content", "")
        return extract_binary(content)
    raise Exception("Rate limited after 4 retries")


def query_gemini(model_id, api_key, prompt, temperature=0.0, max_tokens=512):
    """Google Gemini API (generateContent)."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "text/plain",
        },
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(4):
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        if resp.status_code == 429:
            wait = min(60, 10 * (2 ** attempt))
            print(f"    429 rate limited, waiting {wait}s (attempt {attempt+1})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"].get("message", "Unknown error"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise Exception("No candidates in response")
        parts = candidates[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                return extract_binary(p["text"])
        raise Exception("No text parts in response")
    raise Exception("Rate limited after 4 retries")


def extract_binary(text):
    """Extract 1 or 0 from model output. Handles reasoning models."""
    if not text:
        return None
    text = text.strip()
    # Direct match
    if text in ("1", "0"):
        return int(text)
    # Find last occurrence of standalone 1 or 0
    matches = re.findall(r'\b([01])\b', text)
    if matches:
        return int(matches[-1])
    return None


# ─── Judgment collection ────────────────────────────────────────────────────
def judge_relevance(jd_text, candidate_text, model_name):
    """Return 1 (relevant), 0 (not), or None (error)."""
    prompt = UMBRELA_PROMPT.format(jd_text=jd_text, candidate_text=candidate_text)
    cfg = MODELS[model_name]
    try:
        if cfg["provider"] == "zen":
            return query_zen(cfg["endpoint"], cfg["api_key"], prompt, model_name)
        elif cfg["provider"] == "gemini":
            return query_gemini(cfg["model_id"], cfg["api_key"], prompt)
        return None
    except Exception as e:
        print(f"  ERROR [{model_name}]: {e}")
        return None


# ─── Pool construction ──────────────────────────────────────────────────────
def build_pool(project_root, jd_config_path, n_candidates=500):
    """Pool top-k from: our pipeline + BM25-only + random fill."""
    project_root = Path(project_root)
    jd_config_path = Path(jd_config_path)

    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_input = json.load(f)
    parsed_jd = parse_job_description(jd_input)

    all_cands = []
    for cand in stream_candidates(str(project_root / "candidates.jsonl")):
        all_cands.append(cand)

    bm25 = BM25Filter(all_cands)
    bm25_top = bm25.filter_candidates(parsed_jd, top_n=n_candidates)
    bm25_ids = {c["candidate_id"] for c in bm25_top}

    sub_path = project_root / "outputs" / "mohd_ibadullah.csv"
    our_top_ids = set()
    if sub_path.exists():
        df = pd.read_csv(sub_path)
        our_top_ids = set(df["candidate_id"].tolist()[:n_candidates])

    pool_ids = our_top_ids | bm25_ids
    all_ids = {c["candidate_id"] for c in all_cands}
    remaining = all_ids - pool_ids
    need = 2 * n_candidates - len(pool_ids)
    pool_ids |= set(random.sample(list(remaining), min(need, len(remaining))))

    records = []
    for cid in pool_ids:
        cand = next(c for c in all_cands if c["candidate_id"] == cid)
        profile = cand.get("profile") or {}
        title = profile.get("current_title", "") or ""
        headline = profile.get("headline", "") or ""
        summary = profile.get("summary", "") or ""
        skills = cand.get("skills") or []
        skills_str = ", ".join([s.get("name", "") for s in skills[:15] if isinstance(s, dict)])
        history = cand.get("career_history") or []
        career_titles = " | ".join([r.get("title", "") for r in history[:5] if isinstance(r, dict)])
        candidate_text = (
            f"Title: {title}. Headline: {headline}. Skills: {skills_str}. "
            f"Career: {career_titles}. Summary: {summary}"
        )
        jd_text = (
            f"{parsed_jd['role_title']}. "
            f"Required skills: {', '.join(parsed_jd['required_skills'])}. "
            f"Nice to have: {', '.join(parsed_jd['nice_to_have_skills'])}. "
            f"Domain: {', '.join(parsed_jd.get('domain_keywords', []))}. "
            f"Seniority: {parsed_jd.get('seniority_level', 'senior')}"
        )
        records.append({"candidate_id": cid, "jd_text": jd_text, "candidate_text": candidate_text})

    return records


# ─── κ + bootstrap ──────────────────────────────────────────────────────────
def compute_kappa_and_ci(human_labels, llm_labels):
    """Cohen's kappa with paired bootstrap CI (B=10000)."""
    human = np.array(human_labels, dtype=int)
    llm = np.array(llm_labels, dtype=int)
    n = len(human)

    po = (human == llm).mean()
    p_h1 = (human == 1).mean()
    p_l1 = (llm == 1).mean()
    pe = p_h1 * p_l1 + (1 - p_h1) * (1 - p_l1)
    kappa = (po - pe) / (1 - pe) if 1 - pe != 0 else 0.0

    rng = np.random.default_rng(42)
    boots = []
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        hs, ls = human[idx], llm[idx]
        _ph, _pl = (hs == 1).mean(), (ls == 1).mean()
        _po = (hs == ls).mean()
        _pe = _ph * _pl + (1 - _ph) * (1 - _pl)
        _k = (_po - _pe) / (1 - _pe) if 1 - _pe != 0 else 0.0
        boots.append(_k)

    boots = np.array(boots)
    return float(kappa), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building eval pool...")
    pool = build_pool(project_root, "config/job_description.json", n_candidates=500)
    print(f"Pool size: {len(pool)} candidates")

    random.seed(42)
    sample = random.sample(pool, 100)
    print(f"Sampled {len(sample)} pairs for labeling.\n")

    all_results = {}
    for model_name in MODELS:
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        judgments = []
        latencies = []
        errors = 0
        t_start = time.time()

        for i, rec in enumerate(sample):
            t0 = time.time()
            label = judge_relevance(rec["jd_text"], rec["candidate_text"], model_name)
            elapsed = time.time() - t0
            latencies.append(elapsed)

            if label is not None:
                judgments.append(label)
            else:
                errors += 1
                judgments.append(None)

            if (i + 1) % 20 == 0:
                valid_so_far = sum(1 for j in judgments if j is not None)
                print(f"  {i+1}/100 done (valid: {valid_so_far}, errors: {errors})")

            time.sleep(10)

        total_time = time.time() - t_start
        valid = [j for j in judgments if j is not None]
        pos_rate = sum(valid) / len(valid) if valid else 0

        result = {
            "model": model_name,
            "n_total": 100,
            "n_valid": len(valid),
            "n_errors": errors,
            "pos_rate": round(pos_rate, 4),
            "avg_latency_s": round(np.mean(latencies), 3) if latencies else 0,
            "p50_latency_s": round(np.median(latencies), 3) if latencies else 0,
            "p95_latency_s": round(float(np.percentile(latencies, 95)), 3) if latencies else 0,
            "total_time_s": round(total_time, 1),
            "judgments": judgments,
        }
        all_results[model_name] = result

        raw_path = out_dir / f"raw_judgments_{model_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for j, rec in zip(judgments, sample):
                f.write(json.dumps({
                    "candidate_id": rec["candidate_id"],
                    "label": j,
                    "model": model_name,
                }, ensure_ascii=False) + "\n")

        print(f"  Valid: {len(valid)}/100 | Errors: {errors} | Pos rate: {pos_rate:.2%}")
        print(f"  Latency: avg={result['avg_latency_s']}s p50={result['p50_latency_s']}s p95={result['p95_latency_s']}s")
        print(f"  Saved: {raw_path}\n")

    # Cross-model agreement
    print("\n" + "="*60)
    print("CROSS-MODEL AGREEMENT")
    print("="*60)
    model_names = list(all_results.keys())
    agreement = {}
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            j1, j2 = all_results[m1]["judgments"], all_results[m2]["judgments"]
            pairs = [(a, b) for a, b in zip(j1, j2) if a is not None and b is not None]
            if pairs:
                agree = sum(a == b for a, b in pairs) / len(pairs)
                key = f"{m1} vs {m2}"
                agreement[key] = round(agree, 4)
                print(f"  {key}: {agree:.2%}")

    # Save summary
    summary = {
        "models": {m: {k: v for k, v in r.items() if k != "judgments"} for m, r in all_results.items()},
        "agreement": agreement,
    }
    summary_path = out_dir / "eval_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    header = f"{'Model':<35} {'Valid':>5} {'Err':>4} {'Pos%':>6} {'Avg':>6} {'P50':>6} {'P95':>6}"
    print(header)
    print("-" * len(header))
    for m, r in all_results.items():
        print(f"{m:<35} {r['n_valid']:>5} {r['n_errors']:>4} {r['pos_rate']:>5.1%} {r['avg_latency_s']:>5.1f}s {r['p50_latency_s']:>5.1f}s {r['p95_latency_s']:>5.1f}s")


if __name__ == "__main__":
    main()
