import re
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer: lowercases text and extracts alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

def build_candidate_document(candidate: dict) -> str:
    """
    Combine candidate profile fields into a single text document for lexical search.
    """
    profile = candidate.get("profile") or {}
    current_title = profile.get("current_title", "") or ""
    summary = profile.get("summary", "") or ""
    
    skills = candidate.get("skills") or []
    skills_text = " ".join([s.get("name", "") for s in skills if isinstance(s, dict) and s.get("name")])
    
    career_history = candidate.get("career_history") or []
    career_titles = " ".join([role.get("title", "") for role in career_history if isinstance(role, dict) and role.get("title")])
    # Avoid letting career descriptions dominate search, but include them lightly
    career_descs = " ".join([role.get("description", "")[:200] for role in career_history if isinstance(role, dict) and role.get("description")])
    
    # Combine fields
    document_text = f"{current_title} {summary} {skills_text} {career_titles} {career_descs}"
    return document_text

class BM25Filter:
    def __init__(self, candidates_list: list[dict]) -> None:
        self.candidates = candidates_list or []
        self.corpus = [build_candidate_document(c) for c in self.candidates]
        self.tokenized_corpus = [tokenize(doc) for doc in self.corpus]
        total_tokens = sum(len(doc) for doc in self.tokenized_corpus)
        if total_tokens > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    @classmethod
    def from_corpus(cls, corpus: list[str]) -> "BM25Filter":
        """Build a BM25 index from pre-built document strings (memory-efficient path)."""
        inst = cls.__new__(cls)
        inst.candidates = None
        inst.corpus = corpus or []
        inst.tokenized_corpus = [tokenize(doc) for doc in inst.corpus]
        total_tokens = sum(len(doc) for doc in inst.tokenized_corpus)
        if total_tokens > 0:
            inst.bm25 = BM25Okapi(inst.tokenized_corpus)
        else:
            inst.bm25 = None
        return inst

    def _build_query(self, parsed_jd: dict) -> list[str]:
        query_parts = [
            parsed_jd.get("role_title", "") or "",
            *parsed_jd.get("required_skills", []),
            *parsed_jd.get("nice_to_have_skills", []),
            *parsed_jd.get("domain_keywords", []),
        ]
        return tokenize(" ".join(str(q) for q in query_parts))

    def get_top_indices(self, parsed_jd: dict, top_n: int = 3000) -> list[int]:
        """Return indices of top-N candidates by BM25 score (descending)."""
        count = len(self.candidates) if self.candidates is not None else len(self.corpus or [])
        if count == 0:
            return []
        if not self.bm25:
            return list(range(min(top_n, count)))
        tokenized_query = self._build_query(parsed_jd)
        try:
            scores = self.bm25.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return ranked_indices[:top_n]
        except (ZeroDivisionError, Exception):
            return list(range(min(top_n, count)))
        
    def filter_candidates(self, parsed_jd: dict, top_n: int = 3000) -> list[dict]:
        """
        Rank all candidates using BM25 against the Job Description and return the top N.
        """
        if not self.candidates:
            return []
        top_indices = self.get_top_indices(parsed_jd, top_n=top_n)
        scores = [0.0] * len(self.candidates)
        if self.bm25:
            try:
                tokenized_query = self._build_query(parsed_jd)
                scores = self.bm25.get_scores(tokenized_query)
            except Exception:
                pass

        top_candidates = []
        for idx in top_indices:
            cand = self.candidates[idx]
            cand["_bm25_score"] = float(scores[idx]) if idx < len(scores) else 0.0
            top_candidates.append(cand)

        return top_candidates

