# -*- coding: utf-8 -*-
"""
Name-blindness tests on text BUILDERS — the layer that actually feeds embeddings.

Tests three functions directly:
  1. src/candidate_text.py :: build_candidate_embedding_text
  2. src/bm25_filter.py    :: build_candidate_document
  3. src/cross_encoder_reranker.py :: CrossEncoderReranker.build_candidate_text

Each test:
  - Swaps anonymized_name → output must be byte-identical
  - Output must not contain any token of the name (case-insensitive)
  - Repeats for location, current_company, education[].institution, email, phone
"""
import pytest
from src.candidate_text import build_candidate_embedding_text
from src.bm25_filter import build_candidate_document
from src.cross_encoder_reranker import CrossEncoderReranker

# ── Fixture ───────────────────────────────────────────────────────────────────

def _base_candidate(**overrides) -> dict:
    """Return a fully-formed candidate dict. Override any field via kwargs."""
    cand = {
        "candidate_id": "CAND_TEXT_BLIND_001",
        "profile": {
            "anonymized_name": "Priya Sharma",
            "headline": "ML Engineer | NLP, Search",
            "current_title": "Senior ML Engineer",
            "current_company": "Acme Corp",
            "years_of_experience": 7.0,
            "summary": "ML engineer with 7 years in NLP and search systems.",
            "location": "Bangalore, India",
            "country": "India",
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced"},
            {"name": "NLP", "proficiency": "advanced"},
        ],
        "career_history": [
            {"company": "Acme Corp", "title": "ML Engineer", "description": "Built search ranking systems."},
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "CS"},
        ],
    }
    # Apply overrides into nested dicts
    for key, val in overrides.items():
        if key in ("anonymized_name", "location", "current_company", "email", "phone"):
            cand["profile"][key] = val
        elif key == "education_institution":
            cand["education"][0]["institution"] = val
        else:
            cand[key] = val
    return cand


# Name variants — diverse cultural origins
NAMES = ["Priya Sharma", "John Smith", "Chen Wei", "Fatima Al-Rashid"]
LOCATIONS = ["Bangalore, India", "New York, USA", "Tokyo, Japan", "Berlin, Germany"]
COMPANIES = ["Acme Corp", "Google", "TCS", "Unknown Startup"]
INSTITUTIONS = ["IIT Bombay", "MIT", "Tsinghua University", "Local College"]
EMAILS = ["priya@example.com", "john@work.org", "chen@uni.cn", "fatima@mail.com"]
PHONES = ["+91-98765-43210", "+1-555-123-4567", "+81-90-1234-5678", "+49-170-1234567"]


# ── build_candidate_embedding_text ────────────────────────────────────────────

class TestEmbeddingTextBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_output_identical(self, name):
        base = _base_candidate()
        modified = _base_candidate(anonymized_name=name)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)

    @pytest.mark.parametrize("name", NAMES)
    def test_name_not_in_output(self, name):
        cand = _base_candidate(anonymized_name=name)
        text = build_candidate_embedding_text(cand)
        assert name.lower() not in text.lower(), f"Name '{name}' found in embedding text"

    @pytest.mark.parametrize("location", LOCATIONS)
    def test_location_identical(self, location):
        base = _base_candidate()
        modified = _base_candidate(location=location)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)

    @pytest.mark.parametrize("company", COMPANIES)
    def test_company_identical(self, company):
        base = _base_candidate()
        modified = _base_candidate(current_company=company)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)

    @pytest.mark.parametrize("inst", INSTITUTIONS)
    def test_education_identical(self, inst):
        base = _base_candidate()
        modified = _base_candidate(education_institution=inst)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)

    @pytest.mark.parametrize("email", EMAILS)
    def test_email_identical(self, email):
        base = _base_candidate()
        modified = _base_candidate(email=email)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)

    @pytest.mark.parametrize("phone", PHONES)
    def test_phone_identical(self, phone):
        base = _base_candidate()
        modified = _base_candidate(phone=phone)
        assert build_candidate_embedding_text(base) == build_candidate_embedding_text(modified)


# ── build_candidate_document (BM25) ──────────────────────────────────────────

class TestBM25DocumentBlindToName:
    @pytest.mark.parametrize("name", NAMES)
    def test_output_identical(self, name):
        base = _base_candidate()
        modified = _base_candidate(anonymized_name=name)
        assert build_candidate_document(base) == build_candidate_document(modified)

    @pytest.mark.parametrize("name", NAMES)
    def test_name_not_in_output(self, name):
        cand = _base_candidate(anonymized_name=name)
        text = build_candidate_document(cand)
        # Split name into tokens and check each
        for token in name.split():
            assert token.lower() not in text.lower(), f"Name token '{token}' found in BM25 doc"

    @pytest.mark.parametrize("location", LOCATIONS)
    def test_location_identical(self, location):
        base = _base_candidate()
        modified = _base_candidate(location=location)
        assert build_candidate_document(base) == build_candidate_document(modified)

    @pytest.mark.parametrize("company", COMPANIES)
    def test_company_identical(self, company):
        base = _base_candidate()
        modified = _base_candidate(current_company=company)
        assert build_candidate_document(base) == build_candidate_document(modified)

    @pytest.mark.parametrize("inst", INSTITUTIONS)
    def test_education_identical(self, inst):
        base = _base_candidate()
        modified = _base_candidate(education_institution=inst)
        assert build_candidate_document(base) == build_candidate_document(modified)

    @pytest.mark.parametrize("email", EMAILS)
    def test_email_identical(self, email):
        base = _base_candidate()
        modified = _base_candidate(email=email)
        assert build_candidate_document(base) == build_candidate_document(modified)

    @pytest.mark.parametrize("phone", PHONES)
    def test_phone_identical(self, phone):
        base = _base_candidate()
        modified = _base_candidate(phone=phone)
        assert build_candidate_document(base) == build_candidate_document(modified)


# ── CrossEncoderReranker.build_candidate_text ─────────────────────────────────

class TestCrossEncoderTextBlindToName:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        # Don't load the actual model — we only test build_candidate_text

    @pytest.mark.parametrize("name", NAMES)
    def test_output_identical(self, name):
        base = _base_candidate()
        modified = _base_candidate(anonymized_name=name)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)

    @pytest.mark.parametrize("name", NAMES)
    def test_name_not_in_output(self, name):
        cand = _base_candidate(anonymized_name=name)
        text = self.reranker.build_candidate_text(cand)
        for token in name.split():
            assert token.lower() not in text.lower(), f"Name token '{token}' found in CE text"

    @pytest.mark.parametrize("location", LOCATIONS)
    def test_location_identical(self, location):
        base = _base_candidate()
        modified = _base_candidate(location=location)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)

    @pytest.mark.parametrize("company", COMPANIES)
    def test_company_identical(self, company):
        base = _base_candidate()
        modified = _base_candidate(current_company=company)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)

    @pytest.mark.parametrize("inst", INSTITUTIONS)
    def test_education_identical(self, inst):
        base = _base_candidate()
        modified = _base_candidate(education_institution=inst)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)

    @pytest.mark.parametrize("email", EMAILS)
    def test_email_identical(self, email):
        base = _base_candidate()
        modified = _base_candidate(email=email)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)

    @pytest.mark.parametrize("phone", PHONES)
    def test_phone_identical(self, phone):
        base = _base_candidate()
        modified = _base_candidate(phone=phone)
        assert self.reranker.build_candidate_text(base) == self.reranker.build_candidate_text(modified)
