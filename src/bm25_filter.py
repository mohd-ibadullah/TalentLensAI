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
        
    def filter_candidates(self, parsed_jd: dict, top_n: int = 3000) -> list[dict]:
        """
        Rank all candidates using BM25 against the Job Description and return the top N.
        """
        # Build query from JD
        query_parts = []
        query_parts.append(parsed_jd["role_title"])
        query_parts.extend(parsed_jd["required_skills"])
        query_parts.extend(parsed_jd["nice_to_have_skills"])
        query_parts.extend(parsed_jd["domain_keywords"])
        
        query_text = " ".join(query_parts)
        tokenized_query = tokenize(query_text)
        
        # Calculate BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort candidates by score descending
        candidates_with_scores = list(zip(self.candidates, scores))
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N candidates along with their BM25 score
        top_candidates = []
        for i in range(min(top_n, len(candidates_with_scores))):
            cand, score = candidates_with_scores[i]
            # Attach BM25 score to metadata for record keeping
            cand["_bm25_score"] = float(score)
            top_candidates.append(cand)
            
        return top_candidates

