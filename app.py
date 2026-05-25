import streamlit as st
from vector_search import SemanticSearchEngine
import time

# 1. Page Configuration
st.set_page_config(page_title="ArXiv ML Search", page_icon="🔍", layout="centered")

# 2. Cache the ML Model 
@st.cache_resource
def load_engine():
    return SemanticSearchEngine("paper_embeddings.npy", "ai_papers_final.parquet")

with st.spinner("Loading ML Engine & Vector Database..."):
    engine = load_engine()

# 3. UI Layout
st.title("🔍 Neural ArXiv Search Engine")
st.markdown("Search across 425,000+ CS/AI research papers using dense vector embeddings.")

# 4. Search Bar
query = st.text_input("Enter a technical concept:", placeholder="e.g., optimizing LLM inference latency")

if st.button("Search", type="primary"):
    if query:
        start_time = time.time()
        
        # Run the search directly
        results = engine.search(query, top_k=5)
        
        latency = time.time() - start_time
        st.success(f"⚡ Retrieved {len(results)} papers in {latency:.4f} seconds!")
        
        # 5. Display the Results
        for res in results:
            # Fetch the ID from dataframe metadata and construct the URL
            arxiv_url = f"https://arxiv.org/abs/{res['id']}"
            
            st.markdown(f"### [{res['title']}]({arxiv_url})")
            st.caption(f"Rank: {res['rank']} | Similarity Score: {res['score']:.4f} | Categories: {res['categories']}")
            
            st.write(res['abstract'])
            st.divider()