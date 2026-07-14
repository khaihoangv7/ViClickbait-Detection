"""
Training script for PhoBERT + Linguistic Features Fusion model.
Extracts hand-crafted features, concatenates them with PhoBERT CLS outputs, and trains.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import json

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    PHOBERT_MODEL_NAME,
    PHOBERT_CHECKPOINT_DIR,
    FUSION_CHECKPOINT_DIR,
    RESULTS_DIR,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    LABEL_MAP,
    ensure_dirs
)
from src.models.phobert_fusion import PhoBERTFusionClassifier
from src.models.linguistic_features import get_linguistic_features_matrix

class ClickbaitFusionDataset(Dataset):
    """PyTorch Dataset for PhoBERT + Linguistic Features Fusion."""
    def __init__(self, csv_file, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Text and labels
        self.titles = self.df['title_segmented'].fillna("").values
        self.labels = self.df['label'].map(LABEL_MAP).values
        
        # Hand-crafted features (N, 7)
        # Use the raw titles (non-segmented) for linguistic features
        print(f"Extracting linguistic features for {csv_file}...")
        raw_titles = self.df['title'].fillna("").values
        self.ling_features = get_linguistic_features_matrix(raw_titles)
        print(f"Feature matrix shape: {self.ling_features.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        label = self.labels[idx]
        ling_feat = self.ling_features[idx]
        
        encoding = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'linguistic_features': torch.tensor(ling_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    predictions = []
    real_values = []
    
    for batch in tqdm(data_loader, desc="Training Fusion"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ling_feats = batch['linguistic_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_features=ling_feats
            )
            loss = loss_fn(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        real_values.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(data_loader)
    epoch_f1 = f1_score(real_values, predictions, average='macro')
    return avg_loss, epoch_f1

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ling_feats = batch['linguistic_features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_features=ling_feats
            )
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    epoch_f1 = f1_score(real_values, predictions, average='macro')
    return avg_loss, epoch_f1, real_values, predictions

def main():
    ensure_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PhoBERT tokenizer
    print("Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    
    # Create datasets & dataloaders
    print("Preparing Datasets & DataLoaders...")
    train_dataset = ClickbaitFusionDataset(TRAIN_PATH, tokenizer)
    val_dataset = ClickbaitFusionDataset(VAL_PATH, tokenizer)
    test_dataset = ClickbaitFusionDataset(TEST_PATH, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = PhoBERTFusionClassifier()
    
    # Optionally load fine-tuned weights from standard PhoBERT classifier
    best_phobert_path = os.path.join(PHOBERT_CHECKPOINT_DIR, "best_phobert.pt")
    if os.path.exists(best_phobert_path):
        print(f"Loading fine-tuned encoder weights from {best_phobert_path}...")
        phobert_state = torch.load(best_phobert_path, map_location='cpu')
        # Filter state dict for phobert backbone keys only
        backbone_state = {k.replace('phobert.', ''): v for k, v in phobert_state.items() if k.startswith('phobert.')}
        missing, unexpected = model.phobert.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print("No fine-tuned PhoBERT weights found. Training encoder from pretrained backbone.")
        
    model = model.to(device)
    
    # Class weights for imbalance
    class_weights = torch.tensor([1.0, 2.2]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training details
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(FUSION_CHECKPOINT_DIR, "best_fusion.pt")
    
    print("Starting Fusion Model training loop...")
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, scaler)
        val_loss, val_f1, _, _ = eval_model(model, val_loader, loss_fn, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1 Macro: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1 Macro:   {val_f1:.4f}")
        
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        # Checkpoint and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best fusion model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Training stopped.")
                break
                
    # Save training history
    with open(os.path.join(RESULTS_DIR, "fusion_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # Load best checkpoint and evaluate on Test set
    print("\nTraining complete. Evaluating best fusion model on the Test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_f1, y_true, y_pred = eval_model(model, test_loader, loss_fn, device)
    
    print("\n--- Test Set Classification Report (Fusion Model) ---")
    print(classification_report(y_true, y_pred, target_names=LABEL_MAP.keys()))
    
    # Save test results
    report_text = classification_report(y_true, y_pred, target_names=LABEL_MAP.keys())
    
    with open(os.path.join(RESULTS_DIR, "fusion_metrics.txt"), "w") as f:
        f.write("Fusion Classifier Test Metrics:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Test F1 Macro: {test_f1:.4f}\n")
        f.write(f"Test Precision Macro: {precision_score(y_true, y_pred, average='macro'):.4f}\n")
        f.write(f"Test Recall Macro: {recall_score(y_true, y_pred, average='macro'):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_text)
        
    print("Fusion Model training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
