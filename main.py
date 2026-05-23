import data_loader

if __name__ == "__main__":
    # Load and filter arxiv data using the generic loader
    df = data_loader.load_arxiv_data()
    
    print(df.head())
    
    output_file = "filtered_arxiv_papers.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} filtered papers to '{output_file}'")