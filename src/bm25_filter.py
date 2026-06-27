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
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    
    skills = [s.get("name", "") for s in candidate.get("skills", [])]
    skills_text = " ".join(skills)
    
    career_history = candidate.get("career_history", [])
    career_titles = " ".join([role.get("title", "") for role in career_history])
    # Avoid letting career descriptions dominate search, but include them lightly
    career_descs = " ".join([role.get("description", "")[:200] for role in career_history])
    
    # Combine fields
    document_text = f"{current_title} {summary} {skills_text} {career_titles} {career_descs}"
    return document_text

class BM25Filter:
    def __init__(self, candidates_list: list[dict]) -> None:
        self.candidates = candidates_list
        self.corpus = [build_candidate_document(c) for c in candidates_list]
        self.tokenized_corpus = [tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_corpus(cls, corpus: list[str]) -> "BM25Filter":
        """Build a BM25 index from pre-built document strings (memory-efficient path)."""
        inst = cls.__new__(cls)
        inst.candidates = None
        inst.corpus = corpus
        inst.tokenized_corpus = [tokenize(doc) for doc in corpus]
        inst.bm25 = BM25Okapi(inst.tokenized_corpus)
        return inst

    def _build_query(self, parsed_jd: dict) -> list[str]:
        query_parts = [
            parsed_jd["role_title"],
            *parsed_jd["required_skills"],
            *parsed_jd["nice_to_have_skills"],
            *parsed_jd["domain_keywords"],
        ]
        return tokenize(" ".join(query_parts))

    def get_top_indices(self, parsed_jd: dict, top_n: int = 3000) -> list[int]:
        """Return indices of top-N candidates by BM25 score (descending)."""
        tokenized_query = self._build_query(parsed_jd)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:top_n]
        
    def filter_candidates(self, parsed_jd: dict, top_n: int = 3000) -> list[dict]:
        """
        Rank all candidates using BM25 against the Job Description and return the top N.
        """
        top_indices = self.get_top_indices(parsed_jd, top_n=top_n)
        tokenized_query = self._build_query(parsed_jd)
        scores = self.bm25.get_scores(tokenized_query)

        top_candidates = []
        for idx in top_indices:
            cand = self.candidates[idx]
            cand["_bm25_score"] = float(scores[idx])
            top_candidates.append(cand)

        return top_candidates

