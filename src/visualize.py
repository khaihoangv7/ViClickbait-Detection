"""
Visualization module.
Generates publication-quality charts and plots for the LaTeX paper and saves them to paper/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.config import DATASET_PATH, RESULTS_DIR

# Custom visual settings for academic papers
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

# Path to paper figures
FIGURES_DIR = r"d:\FPT\Period7\DBM\Project\paper\figures"

def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_dataset_statistics():
    """Plot label and category distribution of the dataset."""
    if not os.path.exists(DATASET_PATH):
        print("Dataset CSV not found. Skipping dataset stats plot.")
        return
        
    df = pd.read_csv(DATASET_PATH)
    
    # 1. Label distribution
    plt.figure(figsize=(6, 4))
    colors = ['#1f77b4', '#ff7f0e'] # Factual vs Clickbait
    label_counts = df['label'].value_counts()
    
    # Make pie chart
    plt.pie(
        label_counts, 
        labels=['Factual (Non-clickbait)', 'Clickbait'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        explode=(0, 0.05),
        shadow=True
    )
    plt.title("Label Distribution in ViClickbait-2025")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_distribution.pdf"))
    plt.close()
    
    # 2. Category distribution
    plt.figure(figsize=(10, 5))
    cat_counts = df.groupby(['category', 'label']).size().unstack(fill_value=0)
    # Sort by total count
    cat_counts['total'] = cat_counts.sum(axis=1)
    cat_counts = cat_counts.sort_values(by='total', ascending=False).drop(columns='total')
    
    cat_counts.plot(kind='bar', stacked=True, color=colors, figsize=(10, 5))
    plt.title("Clickbait Distribution by News Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45, ha='right')
    plt.legend(["Factual", "Clickbait"], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "category_distribution.pdf"))
    plt.close()
    print("Dataset distribution plots generated!")

def plot_confusion_matrices():
    """Plot confusion matrices side by side for SVM, PhoBERT, and Fusion models."""
    svm_path = os.path.join(RESULTS_DIR, "svm_cm.npy")
    phobert_path = os.path.join(RESULTS_DIR, "phobert_cm.npy")
    fusion_path = os.path.join(RESULTS_DIR, "fusion_cm.npy")
    
    if not (os.path.exists(svm_path) and os.path.exists(phobert_path) and os.path.exists(fusion_path)):
        print("Confusion matrices not found. Skipping plot.")
        return
        
    svm_cm = np.load(svm_path)
    phobert_cm = np.load(phobert_path)
    fusion_cm = np.load(fusion_path)
    
    cms = [svm_cm, phobert_cm, fusion_cm]
    titles = ["TF-IDF + SVM", "Fine-tuned PhoBERT", "PhoBERT + Fusion (Proposed)"]
    labels = ["Non-Clickbait", "Clickbait"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for idx, (cm, title) in enumerate(zip(cms, titles)):
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            ax=axes[idx],
            xticklabels=labels, 
            yticklabels=labels,
            cbar=False,
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        axes[idx].set_title(title, fontsize=13)
        axes[idx].set_xlabel("Predicted Label")
        if idx == 0:
            axes[idx].set_ylabel("True Label")
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    print("Confusion matrices heatmaps generated!")

def plot_model_comparison():
    """Plot model comparison bar charts."""
    comp_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(comp_csv):
        print("Model comparison CSV not found. Skipping plot.")
        return
        
    df = pd.read_csv(comp_csv)
    
    # Set metrics to plot
    metrics = ['Accuracy', 'F1 Macro', 'F1 Clickbait']
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_melted, 
        x='Metric', 
        y='Score', 
        hue='Model', 
        palette='Set2'
    )
    plt.ylim(0.65, 1.0)
    plt.title("Performance Comparison Across Models")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.pdf"))
    plt.close()
    print("Model comparison chart generated!")

def plot_training_curves():
    """Plot PhoBERT and Fusion loss training curves."""
    pb_history_path = os.path.join(RESULTS_DIR, "phobert_history.json")
    fs_history_path = os.path.join(RESULTS_DIR, "fusion_history.json")
    
    has_pb = os.path.exists(pb_history_path)
    has_fs = os.path.exists(fs_history_path)
    
    # Defaults
    pb_epochs = list(range(1, 6))
    fs_epochs = list(range(1, 6))
    pb_loss = [0.65, 0.45, 0.32, 0.25, 0.18]
    pb_val_loss = [0.55, 0.38, 0.30, 0.28, 0.27]
    fs_loss = [0.58, 0.38, 0.28, 0.20, 0.14]
    fs_val_loss = [0.48, 0.33, 0.26, 0.23, 0.22]
    
    if has_pb:
        with open(pb_history_path, 'r') as f:
            pb_data = json.load(f)
        pb_epochs = list(range(1, len(pb_data['train_loss']) + 1))
        pb_loss = pb_data['train_loss']
        pb_val_loss = pb_data['val_loss']
        
    if has_fs:
        with open(fs_history_path, 'r') as f:
            fs_data = json.load(f)
        fs_epochs = list(range(1, len(fs_data['train_loss']) + 1))
        fs_loss = fs_data['train_loss']
        fs_val_loss = fs_data['val_loss']
            
    # Plot curves
    plt.figure(figsize=(8, 4))
    if has_pb:
        plt.plot(pb_epochs, pb_loss, 'b--', label='PhoBERT Train Loss')
        plt.plot(pb_epochs, pb_val_loss, 'b-', label='PhoBERT Val Loss')
    if has_fs:
        plt.plot(fs_epochs, fs_loss, 'g--', label='Fusion Train Loss')
        plt.plot(fs_epochs, fs_val_loss, 'g-', label='Fusion Val Loss')
        
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Set xticks to be the union of both epoch lists
    all_epochs = sorted(list(set(pb_epochs + fs_epochs)))
    plt.xticks(all_epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_curves.pdf"))
    plt.close()
    print("Training loss curves generated!")


def main():
    ensure_figures_dir()
    plot_dataset_statistics()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_training_curves()

if __name__ == "__main__":
    main()
