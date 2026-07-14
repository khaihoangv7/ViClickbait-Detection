"""
Configuration constants for the Vietnamese Clickbait Detection project.
Contains all paths, hyperparameters, and settings used across the project.
"""

import os

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "ViClickbait-2025")
DATASET_PATH = os.path.join(DATA_DIR, "clickbait_dataset_vietnamese.csv")

# Processed data splits
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# Model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PHOBERT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "phobert")
FUSION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "fusion")
SVM_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "svm")

# Results / logs
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# =============================================================================
# Dataset Info
# =============================================================================
LABEL_MAP = {"non-clickbait": 0, "clickbait": 1}
LABEL_MAP_INV = {0: "non-clickbait", 1: "clickbait"}
NUM_CLASSES = 2

# Text columns used for modelling
TEXT_COLUMN = "title"  # primary text field
LEAD_COLUMN = "lead_paragraph"
LABEL_COLUMN = "label"

# =============================================================================
# Data Split Ratios
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# =============================================================================
# PhoBERT Hyperparameters
# =============================================================================
PHOBERT_MODEL_NAME = "vinai/phobert-base"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 3

# Mixed precision
USE_FP16 = True

# =============================================================================
# TF-IDF + SVM Hyperparameters
# =============================================================================
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

SVM_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

# =============================================================================
# Linguistic Feature Constants
# =============================================================================
VAGUE_PHRASES = [
    "người này",
    "bí mật ấy",
    "sự thật về",
    "điều đó",
    "nơi này",
    "ai đó",
]

EXAGGERATED_PHRASES = [
    "không thể tin nổi",
    "ngỡ ngàng",
    "gây sốc",
    "chấn động",
    "kinh hoàng",
    "bất ngờ",
]

# Number of hand-crafted linguistic features
NUM_LINGUISTIC_FEATURES = 7

# =============================================================================
# Fusion Model Hyperparameters
# =============================================================================
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.3
PHOBERT_HIDDEN_SIZE = 768  # CLS token dimension

# =============================================================================
# Helper: ensure directories exist
# =============================================================================
def ensure_dirs():
    """Create all required output directories if they don't exist."""
    for d in [PROCESSED_DIR, PHOBERT_CHECKPOINT_DIR,
              FUSION_CHECKPOINT_DIR, SVM_CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
