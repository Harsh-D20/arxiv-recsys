import numpy as np
import pandas as pd
import faiss
import torch
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ------------------------------------------------------------
# THE CACHED LLM EXPANSION FUNCTION
# @lru_cache(maxsize=1000) stores the last 1000 searches in RAM. 
# If a user searches the same thing twice, it skips the API call entirely!
# ------------------------------------------------------------
@lru_cache(maxsize=1000)
def expand_query(query: str) -> str:
    """Calls an LLM to generate synonyms and appends them to the query."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No Groq API Key found. Skipping expansion.", flush=True)
        return query
        
    try:
        client = Groq(api_key=api_key)
        
        # We use a very strict prompt so the LLM doesn't get chatty
        prompt = f"""Given this technical search query, generate 3 to 5 related technical synonyms, alternate acronyms, or related core concepts. 
        Return ONLY the words separated by spaces. Do not write a sentence. Do not use commas.
        Query: {query}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # Extremely fast, lightweight model
            temperature=0.1,        # Low temperature for deterministic, factual outputs
            max_tokens=30,
        )
        
        synonyms = chat_completion.choices[0].message.content.strip()
        expanded = f"{query} {synonyms}"
        print(f"\n[CACHE MISS] API Called! Expanded '{query}' -> '{expanded}'", flush=True)
        return expanded
        
    except Exception as e:
        print(f"API Error during expansion: {e}", flush=True)
        return query


class SemanticSearchEngine:
    def __init__(self, embeddings_path, metadata_path):
        print("1. Loading embeddings and metadata...")
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_parquet(metadata_path)
        self.dimension = self.embeddings.shape[1]
        
        print("2. Building FAISS (Dense) Index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        
        print("3. Building BM25 (Sparse) Index...")
        corpus = (self.df['title'] + " " + self.df['abstract']).fillna("").tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("4. Loading embedding model for queries...")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
        print("Hybrid Search Engine Ready!\n")

    def search(self, query, top_k=5):
        # --- NEW: EXPAND THE QUERY ---
        expanded_query = expand_query(query)
        
        # --- 1. DENSE SEARCH (FAISS) ---
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_query = instruction + expanded_query
        query_vector = self.model.encode([formatted_query], normalize_embeddings=True)
        
        dense_distances, dense_indices = self.index.search(query_vector, 50) 
        dense_results = dense_indices[0].tolist()
        
        # --- 2. SPARSE SEARCH (BM25) ---
        tokenized_query = expanded_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_results = np.argsort(bm25_scores)[::-1][:50].tolist()

        # --- 3. RECIPROCAL RANK FUSION (RRF) ---
        rrf_scores = {}
        for rank, row_idx in enumerate(dense_results):
            rrf_scores[row_idx] = rrf_scores.get(row_idx, 0) + (1.0 / (rank + 60))
        for rank, row_idx in enumerate(sparse_results):
            rrf_scores[row_idx] = rrf_scores.get(row_idx, 0) + (1.0 / (rank + 60))
            
        sorted_fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # --- 4. FORMAT RESULTS ---
        results = []
        for rank_idx, (row_idx, fusion_score) in enumerate(sorted_fused_results):
            paper = self.df.iloc[row_idx]
            results.append({
                "rank": rank_idx + 1,
                "score": fusion_score,
                "title": paper['title'],
                "categories": paper['categories'],
                "abstract": paper['abstract'],
                "id": paper['id']
            })
            
        return results, expanded_query

if __name__ == "__main__":
    engine = SemanticSearchEngine("paper_embeddings.npy", "ai_papers_final.parquet")
    
    test_query = "dynamic guardrail models"
    print(f"\nSearching for: '{test_query}'")
    results = engine.search(test_query, top_k=5)
    
    for res in results:
        print(f"{res['rank']}. [Score: {res['score']:.4f}] {res['title']}")