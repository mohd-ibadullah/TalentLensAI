import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

class EmbeddingScorer:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        """
        Load the tokenizer and model locally using Hugging Face transformers.
        BGE-base-en-v1.5 produces 768-dim embeddings with superior retrieval quality
        compared to MiniLM-L6-v2 (384-dim), while staying fast on CPU.
        Requires instruction prefix for queries.
        """
        print(f"Loading embedding model '{model_name}' (CPU)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval() # Set model to evaluation mode
        self.model_name = model_name
        self._is_bge = "bge" in model_name.lower()
        self.candidate_embeddings = None
        self.candidate_ids = []
        self.id_to_index = {}
        
    def load_precomputed_embeddings(self, embeddings_path: str, ids_path: str):
        """
        Loads the precomputed embeddings matrix and Candidate IDs ordering list.
        """
        print(f"Loading precomputed embeddings from {embeddings_path}...")
        self.candidate_embeddings = np.load(embeddings_path) # Shape: (N, D)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.candidate_ids = json.load(f)
        self.id_to_index = {cid: idx for idx, cid in enumerate(self.candidate_ids)}
        print(f"Loaded {len(self.candidate_ids)} candidate vectors of dimensions {self.candidate_embeddings.shape[1]}")

    def search_similar_candidates(self, jd_text: str, top_n: int = 1000) -> list[tuple[str, float]]:
        """
        Computes cosine similarities between the JD and all 100K precomputed candidate vectors.
        Since vectors are L2-normalized, cosine similarity is just the dot product.
        Returns a list of tuples (candidate_id, score) sorted by score descending.
        """
        if self.candidate_embeddings is None:
            raise ValueError("Precomputed embeddings matrix is not loaded. Call load_precomputed_embeddings first.")
            
        # Get query embedding (with instruction prefix if BGE)
        jd_embedding = self.get_embeddings([jd_text], is_query=True)[0] # Shape: (D,)
        
        # Dot product over all 100K candidates matrix
        similarities = np.dot(self.candidate_embeddings, jd_embedding) # Shape: (N,)
        
        # Sort and get top_n
        top_indices = np.argsort(-similarities)[:top_n]
        
        results = []
        for idx in top_indices:
            results.append((self.candidate_ids[idx], float(similarities[idx])))
            
        return results

    def get_candidate_similarity_by_id(self, candidate_id: str, jd_embedding: np.ndarray) -> float:
        """
        Lookup candidate embedding by ID and calculate dot product with query embedding.
        """
        idx = self.id_to_index.get(candidate_id)
        if idx is not None and self.candidate_embeddings is not None:
            cand_embedding = self.candidate_embeddings[idx]
            return float(np.dot(cand_embedding, jd_embedding))
        return 0.5 # Neutral fallback if not found or not loaded

        
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings using the attention mask.
        """
        token_embeddings = model_output[0] # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _cls_pooling(self, model_output):
        """
        Use [CLS] token embedding — the recommended pooling strategy for BGE models.
        """
        return model_output[0][:, 0]  # First token ([CLS]) from each sequence

    def get_embeddings(self, texts: list[str], batch_size: int = 128, is_query: bool = False) -> np.ndarray:
        """
        Computes L2-normalized embeddings for a list of texts in batches.
        For BGE models, query texts get an instruction prefix for better retrieval.
        """
        all_embeddings = []
        
        # BGE models require instruction prefix for query (not for passages/documents)
        if self._is_bge and is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize sentences
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=160,  # Balanced: captures key content while keeping CPU time reasonable
                return_tensors='pt'
            )
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # BGE uses [CLS] pooling; other models use mean pooling
            if self._is_bge:
                batch_embeddings = self._cls_pooling(model_output)
            else:
                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # L2 Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.numpy())
            
        return np.vstack(all_embeddings)

    def compute_similarity(self, jd_text: str, candidate_texts: list[str]) -> list[float]:
        """
        Computes cosine similarity between the JD text and a list of candidate texts.
        Since embeddings are L2 normalized, cosine similarity is just the dot product.
        """
        if not candidate_texts:
            return []
            
        # Compute JD embedding (as query — gets instruction prefix for BGE)
        jd_embedding = self.get_embeddings([jd_text], is_query=True)[0] # Shape: (D,)
        
        # Compute candidate embeddings (as passages — no instruction prefix)
        cand_embeddings = self.get_embeddings(candidate_texts, is_query=False) # Shape: (N, D)
        
        # Compute dot product
        similarities = np.dot(cand_embeddings, jd_embedding) # Shape: (N,)
        
        return list(similarities.astype(float))

