import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(parquet_file: str) -> None:
    print("1. Loading dataset...")
    # Load the subset we created in the last step
    df = pd.read_parquet(parquet_file)
    
    # Optional: If you have 100,000+ papers and a slow laptop, 
    # uncomment the next line to prototype on just 10,000 papers first.
    # df = df.head(10000).copy() 
    
    print(f"Loaded {len(df)} papers.")

    print("2. Preparing text for embedding...")
    # We combine the title and abstract. 
    # A clear title + dense abstract gives the model the best semantic context.
    df['text_to_embed'] = df['title'] + " [SEP] " + df['abstract']
    
    # Determine the best hardware accelerator (CUDA for NVIDIA, MPS for Mac Silicon, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"3. Loading SentenceTransformer model on device: {device}...")
    
    # all-MiniLM-L6-v2 is small (80MB) but highly effective for semantic search
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("4. Generating embeddings (this may take a few minutes depending on your hardware)...")
    # Convert the pandas series to a list of strings
    sentences = df['text_to_embed'].tolist()
    
    # Encode in batches. show_progress_bar=True will give you a nice tqdm loading bar.
    embeddings = model.encode(sentences, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"Successfully generated embeddings of shape: {embeddings.shape}")
    
    print("5. Saving embeddings and metadata...")
    # Save the raw numpy array of embeddings. This format is perfect for FAISS (our vector database)
    np.save("paper_embeddings.npy", embeddings)
    
    # Save the dataframe again (just in case we dropped rows or want to keep the text_to_embed column)
    df.drop(columns=['text_to_embed']).to_parquet("ai_papers_final.parquet", index=False)
    
    print("Done! Saved 'paper_embeddings.npy' and 'ai_papers_final.parquet'.")