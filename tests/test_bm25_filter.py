import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bm25_filter import tokenize, build_candidate_document, BM25Filter

def test_tokenize():
    assert tokenize("Hello World!") == ["hello", "world"]
    assert tokenize("Python, NLP; and AI.") == ["python", "nlp", "and", "ai"]
    assert tokenize("") == []

def test_build_candidate_document():
    candidate = {
        "profile": {
            "current_title": "AI Engineer",
            "summary": "Specialized in vector search and retrieval."
        },
        "skills": [
            {"name": "Python"},
            {"name": "MLOps"}
        ],
        "career_history": [
            {
                "title": "Backend Developer",
                "description": "Built database systems."
            }
        ]
    }
    
    doc = build_candidate_document(candidate)
    assert "AI Engineer" in doc
    assert "vector search" in doc
    assert "Python" in doc
    assert "Backend Developer" in doc

def test_bm25_filter():
    candidates = [
        {
            "candidate_id": "CAND_AI",
            "profile": {"current_title": "Senior AI Engineer", "summary": "Working on retrieval systems and vector database indexing."},
            "skills": [{"name": "Python"}, {"name": "NLP"}],
            "career_history": []
        },
        {
            "candidate_id": "CAND_ACCNT",
            "profile": {"current_title": "Corporate Accountant", "summary": "Expert in auditing and balance sheets."},
            "skills": [{"name": "Accounting"}, {"name": "Excel"}],
            "career_history": []
        },
        {
            "candidate_id": "CAND_OTHER",
            "profile": {"current_title": "Nurse Practitioner", "summary": "Providing patient care in hospital setting."},
            "skills": [{"name": "Nursing"}, {"name": "Healthcare"}],
            "career_history": []
        }
    ]
    
    parsed_jd = {
        "role_title": "AI Engineer",
        "required_skills": ["Python", "NLP"],
        "nice_to_have_skills": [],
        "domain_keywords": ["retrieval", "vector search"]
    }
    
    bm25 = BM25Filter(candidates)
    results = bm25.filter_candidates(parsed_jd, top_n=3)
    
    assert len(results) == 3
    # CAND_AI should rank first because it matches AI Engineer, Python, NLP, and retrieval keywords
    assert results[0]["candidate_id"] == "CAND_AI"
    assert results[0]["_bm25_score"] > results[1]["_bm25_score"]

