import kagglehub
import os
import json
import pandas as pd

DEFAULT_CATEGORIES = {
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language (NLP)
}

def download_arxiv_dataset():
    # Download latest version of arxiv dataset from kaggle
    cache_path = kagglehub.dataset_download("Cornell-University/arxiv")
    file_path = os.path.join(cache_path, "arxiv-metadata-oai-snapshot.json")
    return file_path

def stream_arxiv_data(file_path, target_categories=None):
    """
    A generator that yields one paper at a time.
    This prevents Out-Of-Memory (OOM) errors by never loading the whole 3.5GB file into RAM.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            
            # If target categories are provided, filter by them
            if target_categories:
                # arXiv categories are space-separated strings (e.g., "cs.AI cs.LG")
                paper_categories = set(paper['categories'].split())
                if not paper_categories.intersection(target_categories):
                    continue
            
            # Yield only the fields we care about for Semantic Search
            yield {
                'id': paper['id'],
                'title': paper['title'].replace('\n', ' ').strip(),
                'abstract': paper['abstract'].replace('\n', ' ').strip(),
                'categories': paper['categories'],
                'update_date': paper['update_date']
            }

def get_filtered_papers(file_path, target_categories):
    """
    Returns a list of papers that belong to the target categories.
    This is a convenience function that collects all filtered papers into a list.
    Use with caution on large datasets, as it may lead to OOM errors.
    """
    return list(stream_arxiv_data(file_path, target_categories))

def load_arxiv_data(target_categories=DEFAULT_CATEGORIES, as_dataframe=True):
    """
    Downloads the dataset and returns filtered papers.
    """
    path = download_arxiv_dataset()
    papers = get_filtered_papers(path, target_categories)
    
    if as_dataframe:
        return pd.DataFrame(papers)
    return papers