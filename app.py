from fastapi import FastAPI, Query
from vector_search import SemanticSearchEngine
import time

# Initialize the API
app = FastAPI(title="ArXiv Semantic Search Engine", version="1.0")

# We declare the engine globally but instantiate it at startup
engine = None

@app.on_event("startup")
def load_model():
    """Loads the ML model and FAISS index into memory when the server starts."""
    global engine
    print("Initializing ML Engine... (This may take a few seconds)")
    engine = SemanticSearchEngine("paper_embeddings.npy", "ai_papers_final.parquet")
    print("Engine Ready!")

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    return {"status": "Active", "model_loaded": engine is not None}

@app.get("/api/search")
def search_papers(q: str = Query(..., description="Enter your search query"), top_k: int = 5):
    """The main endpoint that connects the web to our ML search engine."""
    start_time = time.time()
    
    # Run the ML search
    results = engine.search(query=q, top_k=top_k)
    
    latency = time.time() - start_time
    
    return {
        "query": q,
        "latency_seconds": round(latency, 4),
        "total_results": len(results),
        "results": results
    }