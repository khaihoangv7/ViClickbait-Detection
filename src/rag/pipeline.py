"""
End-to-End RAG Pipeline for Vietnamese Clickbait Detection and Explanation.
Combines classification (PhoBERT Fusion model), retrieval, and generation.
"""

import os
import torch
from transformers import AutoTokenizer

from src.config import (
    PHOBERT_MODEL_NAME,
    FUSION_CHECKPOINT_DIR,
    LABEL_MAP_INV,
    MAX_SEQ_LENGTH
)
from src.models.phobert_fusion import PhoBERTFusionClassifier
from src.models.linguistic_features import extract_title_features
from src.rag.retriever import ArticleRetriever
from src.rag.generator import ClickbaitExplainer

class VietnameseClickbaitRAGPipeline:
    """
    Unified Pipeline:
    1. Classifies title (Clickbait / Non-clickbait) using the Fusion model.
    2. If Clickbait, retrieves article context and generates explanation + objective title.
    """
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        print(f"Initializing RAG Pipeline on device: {self.device}")
        
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        
        # 2. Load Classifier Model
        self.model_path = os.path.join(FUSION_CHECKPOINT_DIR, "best_fusion.pt")
        self.model = None
        
        if os.path.exists(self.model_path):
            print(f"Loading trained Fusion Model from {self.model_path}...")
            self.model = PhoBERTFusionClassifier()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            print("WARNING: Trained Fusion Model checkpoint not found. Pipeline will use rule-based fallback classification.")
            
        # 3. Load Retriever & Explainer
        self.retriever = ArticleRetriever()
        self.explainer = ClickbaitExplainer()

    def classify_rule_based(self, title):
        """Rule-based classifier fallback."""
        title_lower = title.lower()
        has_question = '?' in title
        has_exclamation = '!' in title
        
        vague_phrases = ["người này", "bí mật ấy", "sự thật về", "điều đó", "nơi này", "ai đó"]
        exag_phrases = ["không thể tin nổi", "ngỡ ngàng", "gây sốc", "chấn động", "kinh hoàng", "bất ngờ"]
        
        has_vague = any(p in title_lower for p in vague_phrases)
        has_exag = any(p in title_lower for p in exag_phrases)
        
        # Classify as clickbait if it matches multiple heuristics
        score = sum([has_question, has_exclamation, has_vague, has_exag])
        if score >= 1:
            return 1, 0.75  # label clickbait, prob 75%
        else:
            return 0, 0.85  # label non-clickbait, prob 85%

    def process(self, raw_title):
        """
        Process a single raw title.
        Returns a dictionary with classification, retrieval, and explanation results.
        """
        # --- Step 1: Classification ---
        if self.model is not None:
            # Word segment the title for PhoBERT
            from src.data_preprocessing import segment_text
            segmented_title = segment_text(raw_title)
            
            # Tokenize
            encoding = self.tokenizer(
                segmented_title,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Extract linguistic features
            feats = extract_title_features(raw_title)
            # Normalize length features (matching train script)
            feats[0] = min(feats[0] / 200.0, 1.0)
            feats[1] = min(feats[1] / 40.0, 1.0)
            feats[2] = min(feats[2] / 5.0, 1.0)
            feats[3] = min(feats[3] / 5.0, 1.0)
            
            ling_feats = torch.tensor([feats], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, ling_feats)
                probs = torch.softmax(logits, dim=1).squeeze()
                pred_label = torch.argmax(probs).item()
                confidence = probs[pred_label].item()
        else:
            # Use rule-based classifier if model is not trained yet
            pred_label, confidence = self.classify_rule_based(raw_title)
            
        label_name = LABEL_MAP_INV[pred_label]
        
        result = {
            "title": raw_title,
            "prediction": label_name,
            "confidence": confidence,
            "retrieved_context": None,
            "explanation": None,
            "rewritten_title": None
        }
        
        # --- Step 2: RAG Explanation (only if classified as clickbait) ---
        if label_name == "clickbait":
            # Retrieve context
            context = self.retriever.retrieve(raw_title)
            result["retrieved_context"] = context
            
            # Generate explanation and rewrite
            rag_output = self.explainer.explain_and_rewrite(raw_title, context)
            result["explanation"] = rag_output["explanation"]
            result["rewritten_title"] = rag_output["rewritten_title"]
            
        return result

if __name__ == "__main__":
    # Test RAG Pipeline
    pipeline = VietnameseClickbaitRAGPipeline(use_gpu=False)
    
    # Test case 1: Clickbait
    title_cb = "Không thể tin nổi: Sự thật về người này đã bị phát hiện!"
    print(f"\n--- Testing Clickbait: '{title_cb}' ---")
    res1 = pipeline.process(title_cb)
    print(f"Prediction: {res1['prediction']} (Confidence: {res1['confidence']:.4f})")
    print(f"Retrieved context: {res1['retrieved_context']}")
    print(f"Explanation: {res1['explanation']}")
    print(f"Rewritten Title: {res1['rewritten_title']}")
    
    # Test case 2: Non-Clickbait
    title_ncb = "Báo cáo tình hình phát triển kinh tế xã hội Việt Nam năm 2024."
    print(f"\n--- Testing Non-Clickbait: '{title_ncb}' ---")
    res2 = pipeline.process(title_ncb)
    print(f"Prediction: {res2['prediction']} (Confidence: {res2['confidence']:.4f})")
    print(f"Rewritten Title: {res2['rewritten_title']}")
