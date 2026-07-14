"""
Data Preprocessing module for Vietnamese Clickbait Detection.
Performs text cleaning, word segmentation using underthesea,
and stratified train/val/test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from tqdm import tqdm
import re

from src.config import (
    DATASET_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    TEXT_COLUMN,
    LEAD_COLUMN,
    LABEL_COLUMN,
    ensure_dirs
)

def clean_text(text):
    """Clean and normalize Vietnamese text."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_text(text):
    """Segment Vietnamese words and join syllables with underscores (e.g. 'học sinh' -> 'học_sinh')."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    try:
        # underthesea word_tokenize with format="text" replaces spaces inside words with underscores
        segmented = word_tokenize(cleaned, format="text")
        return segmented
    except Exception as e:
        # Fallback to original text if word segmentation fails
        return cleaned

def preprocess_pipeline():
    """Main preprocessing pipeline."""
    print("Starting data preprocessing pipeline...")
    ensure_dirs()
    
    # Check if raw dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Raw dataset CSV not found at {DATASET_PATH}")
        
    # Load raw dataset
    print(f"Loading raw dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. Total samples: {len(df)}")
    
    # Fill missing text values
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    df[LEAD_COLUMN] = df[LEAD_COLUMN].fillna("")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("non-clickbait")
    
    # Apply word segmentation to title and lead paragraph with a progress bar
    print("Performing word segmentation on titles...")
    tqdm.pandas(desc="Segmenting Titles")
    df['title_segmented'] = df[TEXT_COLUMN].progress_apply(segment_text)
    
    print("Performing word segmentation on lead paragraphs...")
    tqdm.pandas(desc="Segmenting Lead Paragraphs")
    df['lead_segmented'] = df[LEAD_COLUMN].progress_apply(segment_text)
    
    # Clean and keep original fields as well
    df['title_cleaned'] = df[TEXT_COLUMN].apply(clean_text)
    df['lead_cleaned'] = df[LEAD_COLUMN].apply(clean_text)
    
    # Verify label distribution
    print("Label distribution in raw data:")
    print(df[LABEL_COLUMN].value_counts(normalize=True))
    
    # Stratified split: Train (70%), Val (15%), Test (15%)
    # First split into train and temp (val + test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    val_test_ratio_of_temp = VAL_RATIO / temp_ratio
    
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COLUMN]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_test_ratio_of_temp,
        random_state=RANDOM_SEED,
        stratify=temp_df[LABEL_COLUMN]
    )
    
    print(f"Split results:")
    print(f" - Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f" - Val samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f" - Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save to CSV files
    print(f"Saving splits to {os.path.dirname(TRAIN_PATH)}...")
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print("Preprocessing successfully completed!")

if __name__ == "__main__":
    preprocess_pipeline()
