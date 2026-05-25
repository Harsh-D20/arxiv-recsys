import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import time

class SemanticSearchEngine:
    def __init__(self, embeddings_path, metadata_path):
        print("1. Loading embeddings and metadata...")
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_parquet(metadata_path)
        self.dimension = self.embeddings.shape[1]
        
        print("2. Building FAISS (Dense) Index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        print(f"Dense Index built with {self.index.ntotal} vectors.")
        
        print("3. Building BM25 (Sparse) Index...")
        # We combine title and abstract, lowercase it, and split into words for the exact-match engine
        corpus = (self.df['title'] + " " + self.df['abstract']).fillna("").tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Sparse Index built.")

        print("4. Loading embedding model for queries...")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
        print("Hybrid Search Engine Ready!\n")

    def search(self, query, top_k=5):
        """Runs both searches and merges them using Reciprocal Rank Fusion (RRF)."""
        
        # --- 1. DENSE SEARCH (FAISS) ---
        query_vector = self.model.encode([query], normalize_embeddings=True)
        dense_distances, dense_indices = self.index.search(query_vector, 50) # Pull top 50 to fuse
        dense_results = dense_indices[0].tolist()
        
        # --- 2. SPARSE SEARCH (BM25) ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get the row indices of the top 50 highest scoring BM25 papers
        sparse_results = np.argsort(bm25_scores)[::-1][:50].tolist()

        # --- 3. RECIPROCAL RANK FUSION (RRF) ---
        # Formula: RRF_Score = 1 / (rank + k) -> we use k=60 as industry standard
        rrf_scores = {}
        
        # Score the dense results
        for rank, row_idx in enumerate(dense_results):
            rrf_scores[row_idx] = rrf_scores.get(row_idx, 0) + (1.0 / (rank + 60))
            
        # Score the sparse results
        for rank, row_idx in enumerate(sparse_results):
            rrf_scores[row_idx] = rrf_scores.get(row_idx, 0) + (1.0 / (rank + 60))
            
        # Sort by highest RRF score and grab the top_k
        sorted_fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # --- 4. FORMAT RESULTS ---
        results = []
        for rank_idx, (row_idx, fusion_score) in enumerate(sorted_fused_results):
            paper = self.df.iloc[row_idx]
            results.append({
                "rank": rank_idx + 1,
                "score": fusion_score,  # Now returning the Fusion score!
                "title": paper['title'],
                "categories": paper['categories'],
                "abstract": paper['abstract'],
                "id": paper['id']
            })
            
        return results

if __name__ == "__main__":
    engine = SemanticSearchEngine("paper_embeddings.npy", "ai_papers_final.parquet")
    
    test_query = "dynamic guardrail models"
    print(f"\nSearching for: '{test_query}'")
    results = engine.search(test_query, top_k=5)
    
    for res in results:
        print(f"{res['rank']}. [Score: {res['score']:.4f}] {res['title']}")