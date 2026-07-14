"""
TF-IDF + SVM Baseline Model for Vietnamese Clickbait Detection.
Includes model training, evaluation, and saving checkpoints.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    SVM_CHECKPOINT_DIR,
    RESULTS_DIR,
    LABEL_MAP,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    ensure_dirs
)

def load_data():
    """Load train, val, test CSV datasets. Run preprocessing if missing."""
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH) and os.path.exists(TEST_PATH)):
        print("Processed splits not found. Running data preprocessing...")
        from src.data_preprocessing import preprocess_pipeline
        preprocess_pipeline()
        
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df

def train_eval_svm():
    """Train SVM and evaluate on Validation and Test splits."""
    ensure_dirs()
    train_df, val_df, test_df = load_data()
    
    # Fill any NaNs
    train_df['title_segmented'] = train_df['title_segmented'].fillna("")
    val_df['title_segmented'] = val_df['title_segmented'].fillna("")
    test_df['title_segmented'] = test_df['title_segmented'].fillna("")
    
    # Map label strings to integers
    y_train = train_df['label'].map(LABEL_MAP).values
    y_val = val_df['label'].map(LABEL_MAP).values
    y_test = test_df['label'].map(LABEL_MAP).values
    
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE
    )
    
    X_train = vectorizer.fit_transform(train_df['title_segmented'])
    X_val = vectorizer.transform(val_df['title_segmented'])
    X_test = vectorizer.transform(test_df['title_segmented'])
    
    print(f"TF-IDF Matrix shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Instantiate SVM (using linear kernel, C=1.0 as standard strong baseline)
    print("Training SVM classifier...")
    svm_clf = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    
    # Save the vectorizer and classifier
    vec_path = os.path.join(SVM_CHECKPOINT_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(SVM_CHECKPOINT_DIR, "svm_classifier.pkl")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(svm_clf, model_path)
    print(f"Saved TF-IDF Vectorizer to {vec_path}")
    print(f"Saved SVM Classifier to {model_path}")
    
    # Predict and evaluate
    print("\n--- Validation Results ---")
    val_preds = svm_clf.predict(X_val)
    print(classification_report(y_val, val_preds, target_names=LABEL_MAP.keys()))
    
    print("\n--- Test Results ---")
    test_preds = svm_clf.predict(X_test)
    print(classification_report(y_test, test_preds, target_names=LABEL_MAP.keys()))
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    precision = precision_score(y_test, test_preds, average='macro')
    recall = recall_score(y_test, test_preds, average='macro')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Macro: {f1:.4f}")
    
    # Save metrics to results dir
    metrics_path = os.path.join(RESULTS_DIR, "svm_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"TF-IDF + SVM baseline metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1:.4f}\n")
        f.write(f"Precision Macro: {precision:.4f}\n")
        f.write(f"Recall Macro: {recall:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, test_preds, target_names=LABEL_MAP.keys()))
        
    return vectorizer, svm_clf

if __name__ == "__main__":
    train_eval_svm()
