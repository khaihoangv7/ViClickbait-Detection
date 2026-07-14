"""
PhoBERT Sequence Classifier for Vietnamese Clickbait Detection.
Includes model definition and forward pass.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import PHOBERT_MODEL_NAME, PHOBERT_HIDDEN_SIZE, NUM_CLASSES

class PhoBERTClassifier(nn.Module):
    """
    Standard PhoBERT classifier.
    Extracts the [CLS] representation and feeds it into a classification head.
    """
    def __init__(self, pretrained_model_name=PHOBERT_MODEL_NAME, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super(PhoBERTClassifier, self).__init__()
        
        print(f"Loading pretrained PhoBERT backbone: {pretrained_model_name}...")
        self.phobert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(PHOBERT_HIDDEN_SIZE, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """
        # PhoBERT returns outputs containing last_hidden_state and pooler_output
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (index 0)
        # Shape: (batch_size, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        
        # Linear classifier
        # Shape: (batch_size, num_classes)
        logits = self.classifier(cls_output)
        
        return logits

if __name__ == "__main__":
    # Test model initialization and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoBERTClassifier().to(device)
    print("Model initialized successfully!")
    
    # Fake batch
    batch_size = 2
    seq_len = 10
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        out = model(dummy_input_ids, dummy_mask)
    print("Output logits shape:", out.shape)
    print("Output logits:\n", out)
