"""Tests for src/data_integrity.py — the corpus integrity guard."""
import json
import os
import tempfile

import pytest

from src.data_integrity import check_corpus


# ── Helpers ──────────────────────────────────────────────────────────────────

def _real_record(idx: int = 0) -> dict:
    """Return a fully-populated candidate record (mimics real data)."""
    return {
        "candidate_id": f"CAND_{idx:07d}",
        "profile": {
            "anonymized_name": f"Candidate {idx}",
            "headline": "Backend Engineer | Python, Go",
            "current_title": "Backend Engineer",
            "current_company": "Acme Corp",
            "years_of_experience": 5.0 + (idx % 10),
            "summary": (
                "Experienced backend engineer with strong skills in Python, Go, "
                "and distributed systems. Built microservices serving millions of "
                "requests per day. Comfortable with Kubernetes, Docker, and CI/CD."
            ),
            "location": "Bangalore, India",
            "country": "India",
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced"},
            {"name": "Go", "proficiency": "intermediate"},
            {"name": "PostgreSQL", "proficiency": "advanced"},
            {"name": "Redis", "proficiency": "intermediate"},
            {"name": "Kubernetes", "proficiency": "intermediate"},
            {"name": "Docker", "proficiency": "advanced"},
            {"name": "REST APIs", "proficiency": "advanced"},
            {"name": "Microservices", "proficiency": "advanced"},
            {"name": "CI/CD", "proficiency": "intermediate"},
            {"name": "System Design", "proficiency": "advanced"},
        ],
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Backend Engineer",
                "description": "Built scalable microservices.",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
            }
        ],
    }


def _shell_record(idx: int = 0) -> dict:
    """Return an empty shell record (the corrupted pattern)."""
    return {
        "candidate_id": f"CAND_{idx:07d}",
        "profile": {
            "anonymized_name": "",
            "headline": "",
            "current_title": "",
            "current_company": "",
            "years_of_experience": 0,
            "summary": "",
            "location": "",
            "country": "",
        },
        "skills": [],
        "career_history": [],
        "education": [],
    }


def _write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCheckCorpus:
    def test_healthy_corpus_ok(self, tmp_path):
        """100 real records → ok=True, populated_fraction=1.0."""
        records = [_real_record(i) for i in range(100)]
        path = str(tmp_path / "good.jsonl")
        _write_jsonl(records, path)

        result = check_corpus(path)
        assert result["ok"] is True
        assert result["n_lines"] == 100
        assert result["n_populated"] == 100
        assert result["populated_fraction"] == 1.0
        assert len(result["problems"]) == 0
        assert result["sha256"] != ""

    def test_corrupted_corpus_not_ok(self, tmp_path):
        """10 real + 90 shells → ok=False, populated_fraction=0.10."""
        records = [_real_record(i) for i in range(10)]
        records += [_shell_record(i + 10) for i in range(90)]
        path = str(tmp_path / "corrupted.jsonl")
        _write_jsonl(records, path)

        result = check_corpus(path)
        assert result["ok"] is False
        assert result["n_lines"] == 100
        assert result["n_populated"] == 10
        assert abs(result["populated_fraction"] - 0.10) < 1e-9
        assert any("populated_fraction" in p for p in result["problems"])

    def test_median_line_bytes_flag(self, tmp_path):
        """All shells → median_line_bytes < 500 → flagged as problem."""
        records = [_shell_record(i) for i in range(50)]
        path = str(tmp_path / "all_shells.jsonl")
        _write_jsonl(records, path)

        result = check_corpus(path)
        assert result["ok"] is False
        assert result["median_line_bytes"] < 500
        assert any("median_line_bytes" in p for p in result["problems"])

    def test_mixed_corpus_custom_threshold(self, tmp_path):
        """80 real + 20 shells, threshold 0.75 → ok=True."""
        records = [_real_record(i) for i in range(80)]
        records += [_shell_record(i + 80) for i in range(20)]
        path = str(tmp_path / "mixed.jsonl")
        _write_jsonl(records, path)

        result = check_corpus(path, min_populated_fraction=0.75)
        assert result["ok"] is True
        assert abs(result["populated_fraction"] - 0.80) < 1e-9

    def test_file_not_found(self):
        """Non-existent file → ok=False with FileNotFoundError message."""
        result = check_corpus("/nonexistent/corpus.jsonl")
        assert result["ok"] is False
        assert any("not found" in p.lower() or "not found" in p for p in result["problems"])

    def test_json_parse_error_flagged(self, tmp_path):
        """One malformed JSON line → problem includes parse error count."""
        path = str(tmp_path / "bad_json.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_real_record(0)) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_real_record(1)) + "\n")

        result = check_corpus(path)
        assert any("parse" in p.lower() for p in result["problems"])

    def test_empty_file(self):
        """Empty file → n_lines=0, ok=False."""
        result = check_corpus("/dev/null")
        # On Linux /dev/null is empty; on Windows this path won't exist
        # but the logic still returns ok=False
        assert result["ok"] is False
