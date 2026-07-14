"""
PhoBERT + Linguistic Features Fusion Classifier for Vietnamese Clickbait Detection.
Concatenates semantic features (PhoBERT CLS token) with surface linguistic features.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import (
    PHOBERT_MODEL_NAME, 
    PHOBERT_HIDDEN_SIZE, 
    NUM_LINGUISTIC_FEATURES, 
    FUSION_HIDDEN_DIM, 
    FUSION_DROPOUT, 
    NUM_CLASSES
)

class PhoBERTFusionClassifier(nn.Module):
    """
    Fusion Classifier.
    Concatenates the PhoBERT [CLS] token representation (768-dim) 
    with 7 hand-crafted linguistic features (7-dim) and passes 
    them through a multi-layer classification head.
    """
    def __init__(self, pretrained_model_name=PHOBERT_MODEL_NAME, 
                 num_classes=NUM_CLASSES, 
                 linguistic_dim=NUM_LINGUISTIC_FEATURES,
                 hidden_dim=FUSION_HIDDEN_DIM,
                 dropout_rate=FUSION_DROPOUT):
        super(PhoBERTFusionClassifier, self).__init__()
        
        print(f"Loading pretrained PhoBERT backbone for Fusion: {pretrained_model_name}...")
        self.phobert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Combined size: 768 + 7 = 775
        combined_dim = PHOBERT_HIDDEN_SIZE + linguistic_dim
        
        # Classification head with a hidden layer
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, linguistic_features):
        """
        Forward pass.
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        linguistic_features: (batch_size, 7)
        """
        # Get PhoBERT CLS representation
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        
        # Concatenate CLS token output with linguistic features
        # Shape: (batch_size, 775)
        fused = torch.cat((cls_output, linguistic_features), dim=1)
        
        # Pass through the classification head
        x = self.fc1(fused)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

if __name__ == "__main__":
    # Test model initialization and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoBERTFusionClassifier().to(device)
    print("Fusion Model initialized successfully!")
    
    # Fake batch
    batch_size = 2
    seq_len = 10
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)
    dummy_ling_feats = torch.randn((batch_size, 7)).to(device)
    
    with torch.no_grad():
        out = model(dummy_input_ids, dummy_mask, dummy_ling_feats)
    print("Output logits shape:", out.shape)
    print("Output logits:\n", out)
