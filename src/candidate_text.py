"""Shared candidate text builder — must match precompute_embeddings.py exactly."""
from __future__ import annotations


def build_candidate_embedding_text(cand: dict) -> str:
    profile = cand.get("profile", {})
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    skills_str = ", ".join([s.get("name", "") for s in cand.get("skills", [])[:15]])
    career_titles = " | ".join([r.get("title", "") for r in cand.get("career_history", [])[:5]])
    return (
        f"Title: {title}. Headline: {headline}. Skills: {skills_str}. "
        f"Career: {career_titles}. Summary: {summary}"
    )
