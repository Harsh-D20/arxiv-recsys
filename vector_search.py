import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
import time

class SemanticSearchEngine:
    def __init__(self, embeddings_path, metadata_path):
        print("1. Loading embeddings and metadata...")
        # Load the saved numpy array and pandas dataframe
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_parquet(metadata_path)
        
        # The dimension is the second value of the shape (384)
        self.dimension = self.embeddings.shape[1]
        
        print("2. Building FAISS Index...")
        # Because we used normalize_embeddings=True in Phase 2, 
        # the Inner Product (FlatIP) is exactly equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add all 425k embeddings to the FAISS index
        self.index.add(self.embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")
        
        print("3. Loading embedding model for queries...")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Search Engine Ready!\n")

    def search(self, query, top_k=5):
        """Embeds the query and returns the top K most relevant papers as a list of dicts."""
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(top_k):
            row_idx = indices[0][i]
            score = float(distances[0][i]) # Convert numpy float to standard Python float for JSON
            paper = self.df.iloc[row_idx]
            
            results.append({
                "rank": i + 1,
                "score": score,
                "title": paper['title'],
                "categories": paper['categories'],
                "abstract": paper['abstract']
            })
            
        return results

if __name__ == "__main__":
    # Initialize the engine
    engine = SemanticSearchEngine("paper_embeddings.npy", "ai_papers_final.parquet")
    
    # Run a test search! 
    # Try changing this query to whatever AI topic interests you.
    test_query = "dependency parsing with LLMs"
    engine.search(test_query, top_k=3)