"""
Retriever module for the RAG pipeline.
Indexes lead paragraphs from the dataset and retrieves the most relevant article context for a given headline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATASET_PATH

class ArticleRetriever:
    """
    Retriever that indexes article lead paragraphs from the dataset.
    Supports both direct title matching (exact lookup) and TF-IDF search.
    """
    def __init__(self, dataset_path=DATASET_PATH):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        print(f"Initializing ArticleRetriever with dataset: {dataset_path}...")
        self.df = pd.read_csv(dataset_path)
        self.df['title'] = self.df['title'].fillna("").astype(str)
        self.df['lead_paragraph'] = self.df['lead_paragraph'].fillna("").astype(str)
        
        # Create a dictionary for exact lookups
        # Normalize titles for matching (strip, lowercase)
        self.title_to_lead = {}
        for _, row in self.df.iterrows():
            norm_title = self._normalize_text(row['title'])
            if norm_title and row['lead_paragraph'].strip():
                self.title_to_lead[norm_title] = row['lead_paragraph']
                
        # Index lead paragraphs for TF-IDF search
        self.leads = self.df['lead_paragraph'].values
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.lead_vectors = self.vectorizer.fit_transform(self.leads)
        
    def _normalize_text(self, text):
        return " ".join(text.lower().split())
        
    def retrieve(self, query_title, top_k=1):
        """
        Retrieve the most relevant lead paragraph for the query title.
        First attempts exact matching, then falls back to TF-IDF cosine similarity search.
        """
        norm_query = self._normalize_text(query_title)
        
        # 1. Exact Match Lookup
        if norm_query in self.title_to_lead:
            # print("Exact match found in dataset!")
            return self.title_to_lead[norm_query]
            
        # 2. Vector Search Fallback
        query_vec = self.vectorizer.transform([query_title])
        similarities = cosine_similarity(query_vec, self.lead_vectors).flatten()
        best_indices = np.argsort(similarities)[::-1]
        
        # Get the top lead paragraphs that are not empty
        retrieved_leads = []
        for idx in best_indices:
            lead = self.leads[idx].strip()
            if lead:
                retrieved_leads.append(lead)
                if len(retrieved_leads) >= top_k:
                    break
                    
        if retrieved_leads:
            return retrieved_leads[0]
        else:
            return "Nội dung bài báo đang được cập nhật."

if __name__ == "__main__":
    retriever = ArticleRetriever()
    
    # Test direct match
    test_title = retriever.df['title'].iloc[0]
    retrieved = retriever.retrieve(test_title)
    print(f"Query Title: {test_title}")
    print(f"Retrieved Lead Paragraph (exact match):\n{retrieved}\n")
    
    # Test vector search
    query = "đại dịch covid tại việt nam diễn biến phức tạp"
    retrieved_search = retriever.retrieve(query)
    print(f"Search Query: {query}")
    print(f"Retrieved Lead Paragraph (search match):\n{retrieved_search}\n")
