"""Shared candidate text builder — must match precompute_embeddings.py exactly."""
from __future__ import annotations


def build_candidate_embedding_text(cand: dict) -> str:
    profile = cand.get("profile") or {}
    title = profile.get("current_title", "") or ""
    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    skills = cand.get("skills") or []
    skills_str = ", ".join([s.get("name", "") for s in skills[:15] if isinstance(s, dict)])
    history = cand.get("career_history") or []
    career_titles = " | ".join([r.get("title", "") for r in history[:5] if isinstance(r, dict)])
    return (
        f"Title: {title}. Headline: {headline}. Skills: {skills_str}. "
        f"Career: {career_titles}. Summary: {summary}"
    )
