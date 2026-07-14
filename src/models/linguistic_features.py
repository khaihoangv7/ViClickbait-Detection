"""
Linguistic Feature Extraction module for Vietnamese Clickbait headlines.
Extracts 7 surface-level hand-crafted linguistic features.
"""

import re
import numpy as np
import pandas as pd
from src.config import VAGUE_PHRASES, EXAGGERATED_PHRASES

def extract_title_features(title):
    """
    Extract 7 surface linguistic features from a single title string.
    
    Features:
    1. Title length (characters)
    2. Word count
    3. Number of question marks (?)
    4. Number of exclamation marks (!)
    5. Presence of digits (0 or 1)
    6. Presence of vague words/phrases (0 or 1)
    7. Presence of exaggerated words/phrases (0 or 1)
    """
    if not isinstance(title, str):
        title = ""
        
    title_lower = title.lower()
    
    # 1. Title length (characters)
    char_len = len(title)
    
    # 2. Word count (splitting by space)
    word_count = len(title.split())
    
    # 3. Number of question marks
    q_marks = title.count('?')
    
    # 4. Number of exclamation marks
    ex_marks = title.count('!')
    
    # 5. Presence of digits
    has_digit = 1 if any(char.isdigit() for char in title) else 0
    
    # 6. Presence of vague words/phrases
    has_vague = 0
    for phrase in VAGUE_PHRASES:
        if phrase in title_lower:
            has_vague = 1
            break
            
    # 7. Presence of exaggerated words/phrases
    has_exaggerated = 0
    for phrase in EXAGGERATED_PHRASES:
        if phrase in title_lower:
            has_exaggerated = 1
            break
            
    return [
        float(char_len),
        float(word_count),
        float(q_marks),
        float(ex_marks),
        float(has_digit),
        float(has_vague),
        float(has_exaggerated)
    ]

def get_linguistic_features_matrix(df_column):
    """
    Extract linguistic features for a Pandas Series of title strings.
    Returns a normalized float numpy array of shape (N, 7).
    """
    features = []
    for title in df_column:
        features.append(extract_title_features(title))
    
    features_array = np.array(features, dtype=np.float32)
    
    # Simple Min-Max normalization for length features (char_len, word_count) to keep values stable.
    # char_len is features_array[:, 0], word_count is features_array[:, 1]
    # We use reasonable bounds instead of dataset-dependent min/max to ensure consistency.
    features_array[:, 0] = np.clip(features_array[:, 0] / 200.0, 0.0, 1.0)
    features_array[:, 1] = np.clip(features_array[:, 1] / 40.0, 0.0, 1.0)
    # Question and exclamation marks capped at 5
    features_array[:, 2] = np.clip(features_array[:, 2] / 5.0, 0.0, 1.0)
    features_array[:, 3] = np.clip(features_array[:, 3] / 5.0, 0.0, 1.0)
    
    return features_array

if __name__ == "__main__":
    # Test feature extraction
    sample_title = "Không thể tin nổi: Cô gái ấy đã tiết lộ bí mật động trời về người này!"
    feats = extract_title_features(sample_title)
    print(f"Sample Title: {sample_title}")
    print("Extracted Features:")
    print(f"1. Character length: {feats[0]}")
    print(f"2. Word count: {feats[1]}")
    print(f"3. Question marks: {feats[2]}")
    print(f"4. Exclamation marks: {feats[3]}")
    print(f"5. Has digits: {feats[4]}")
    print(f"6. Has vague words: {feats[5]}")
    print(f"7. Has exaggerated words: {feats[6]}")
    
    series = pd.Series([sample_title, "Báo cáo tài chính quý 2 năm 2024"])
    matrix = get_linguistic_features_matrix(series)
    print("\nNormalized features matrix shape:", matrix.shape)
    print(matrix)
