# Vietnamese Clickbait Detection with PhoBERT + Linguistic Feature Fusion and RAG

## Project Overview

A complete framework for **detecting**, **explaining**, and **rewriting** Vietnamese clickbait headlines using:
- **Model 1**: TF-IDF + SVM (Baseline)
- **Model 2**: Fine-tuned PhoBERT
- **Model 3**: PhoBERT + Linguistic Feature Fusion (Proposed)
- **RAG Pipeline**: Retrieval-Augmented Generation for explanation and headline rewriting

**Dataset**: [ViClickbait-2025](./ViClickbait-2025/) — 3,414 Vietnamese news articles (68.8% factual, 31.2% clickbait)

---

## Results Summary

| Model | Accuracy | Macro F1 | F1 Clickbait |
|---|---|---|---|
| TF-IDF + SVM | 77.78% | 71.75% | 58.70% |
| Fine-tuned PhoBERT | 81.09% | 78.76% | 71.72% |
| **PhoBERT + Fusion (Proposed)** | **82.46%** | **79.83%** | **72.56%** |

RAG Evaluation: ROUGE-1 = 56.19%, ROUGE-L = 44.84%, BLEU = 0.2230

---

## Project Structure

```
Project/
├── ViClickbait-2025/          # Dataset
│   ├── clickbait_dataset_vietnamese.csv
│   └── ...
├── data/                      # Preprocessed train/val/test splits
├── checkpoints/               # Saved model checkpoints
├── results/                   # Evaluation metrics and predictions
│   ├── svm_metrics.txt
│   ├── phobert_metrics.txt
│   ├── fusion_metrics.txt
│   ├── rag_metrics.txt
│   └── model_comparison.csv
├── paper/                     # LaTeX research paper
│   ├── main.tex               # Main paper (IEEEtran format)
│   ├── references.bib         # Bibliography
│   └── figures/               # Generated plots (PDF/PNG)
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_preprocessing.py  # Text cleaning, word segmentation, splitting
│   ├── models/
│   │   ├── tfidf_svm.py       # Baseline SVM model
│   │   ├── linguistic_features.py  # 7 handcrafted feature extractor
│   │   ├── phobert_classifier.py   # PhoBERT classifier head
│   │   └── phobert_fusion.py       # PhoBERT + Fusion classifier
│   ├── rag/
│   │   ├── retriever.py       # TF-IDF based context retriever
│   │   ├── generator.py       # LLM generator (GPT-4o-mini or rule-based)
│   │   └── pipeline.py        # Full RAG pipeline
│   ├── train_phobert.py       # PhoBERT training script
│   ├── train_fusion.py        # Fusion model training script
│   ├── evaluation.py          # Classification evaluation
│   ├── evaluate_rag.py        # RAG BLEU/ROUGE evaluation
│   └── visualize.py           # Generate plots
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Preprocess Data
```bash
python -m src.data_preprocessing
```

### 2. Train Baseline SVM
```bash
python -m src.models.tfidf_svm
```

### 3. Train PhoBERT
```bash
python -m src.train_phobert
```

### 4. Train Fusion Model
```bash
python -m src.train_fusion
```

### 5. Evaluate All Models
```bash
python -m src.evaluation
```

### 6. Evaluate RAG
```bash
python -m src.evaluate_rag
```

### 7. Generate Visualizations
```bash
python -m src.visualize
```

---

## Compiling the Paper

Requires MiKTeX or TeX Live installed.

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using Overleaf: upload the `paper/` folder to [overleaf.com](https://www.overleaf.com).

---

## RAG with OpenAI GPT

To use the GPT-4o-mini generator instead of the rule-based fallback:
```bash
set OPENAI_API_KEY=your_key_here
python -m src.evaluate_rag
```
