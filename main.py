import data_loader
import generate_embeddings

if __name__ == "__main__":
    OUTPUT_FILE = "filtered_arxiv_papers.parquet"

    # Load and filter arxiv data using the generic loader
    df = data_loader.load_or_cache_arxiv_data(OUTPUT_FILE)
    
    print(df.head())

    generate_embeddings.generate_embeddings(OUTPUT_FILE)
