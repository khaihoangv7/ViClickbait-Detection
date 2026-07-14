"""
Evaluation module for RAG Generation (explanation and headline rewriting).
Computes ROUGE and BLEU scores against a golden evaluation subset of human-written reference headlines.
"""

import os
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.config import RESULTS_DIR, ensure_dirs
from src.rag.pipeline import VietnameseClickbaitRAGPipeline

# Define a golden evaluation set for Vietnamese clickbait rewriting
# Contains: clickbait title, lead paragraph context, and human-written factual reference title.
GOLDEN_EVAL_SET = [
    {
        "title": "Không thể tin nổi: Sự thật về người đàn ông 30 năm không ngủ!",
        "context": "Ông Thái Ngọc (Quảng Nam) nổi tiếng thế giới vì có khả năng thức suốt hơn 30 năm sau một trận sốt cao vào năm 1973. Dù mất ngủ kéo dài, ông vẫn khỏe mạnh và làm việc đồng áng bình thường.",
        "reference": "Người đàn ông ở Quảng Nam có khả năng thức suốt hơn 30 năm mà vẫn khỏe mạnh"
    },
    {
        "title": "Sốc với số tiền khủng ca sĩ A quyên góp từ thiện, ai nghe cũng ngỡ ngàng!",
        "context": "Ca sĩ A vừa công bố bảng sao kê chi tiết số tiền 15 tỷ đồng quyên góp được từ các nhà hảo tâm để hỗ trợ đồng bào vùng lũ miền Trung, khẳng định toàn bộ số tiền đã được giải ngân đúng mục đích.",
        "reference": "Ca sĩ A công bố sao kê và giải ngân 15 tỷ đồng quyên góp từ thiện vùng lũ"
    },
    {
        "title": "Chấn động: Bí mật động trời đằng sau việc đóng cửa nhà hàng nổi tiếng nhất quận 1",
        "context": "Nhà hàng ẩm thực Việt tại trung tâm quận 1 phải dừng hoạt động từ ngày mai do hết hạn hợp đồng thuê mặt bằng và chi phí giá thuê tăng cao, đại diện nhà hàng cho biết chưa có kế hoạch mở lại.",
        "reference": "Nhà hàng nổi tiếng tại quận 1 đóng cửa do hết hạn hợp đồng thuê mặt bằng"
    },
    {
        "title": "Người này đã làm điều gì khiến giá xăng trong nước đột ngột giảm mạnh chiều nay?",
        "context": "Liên Bộ Tài chính - Công Thương vừa công bố điều chỉnh giá bán lẻ xăng dầu từ 15h hôm nay, theo đó giá xăng RON 95 giảm 1.200 đồng/lít nhờ xu hướng giảm chung của giá dầu thế giới.",
        "reference": "Giá xăng RON 95 giảm 1.200 đồng mỗi lít theo xu hướng thế giới từ chiều nay"
    },
    {
        "title": "Bạn sẽ khóc khi biết sự thật về cô bé nghèo bán vé số dạo dưới trời mưa lớn!",
        "context": "Một cô bé 10 tuổi tại Cần Thơ tranh thủ đi bán vé số sau giờ học để phụ giúp mẹ bị bệnh hiểm nghèo. Nhiều người dân đã giúp đỡ và kêu gọi quyên góp hỗ trợ học phí cho em.",
        "reference": "Cô bé 10 tuổi tại Cần Thơ bán vé số phụ giúp mẹ bị bệnh hiểm nghèo được hỗ trợ học phí"
    }
]

def evaluate_rag_pipeline():
    """Evaluate RAG pipeline using BLEU and ROUGE metrics on golden dataset."""
    ensure_dirs()
    print("Initializing RAG pipeline for evaluation...")
    
    # Initialize pipeline (without GPU for evaluation speed)
    pipeline = VietnameseClickbaitRAGPipeline(use_gpu=False)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    chencherry = SmoothingFunction()
    
    r1_scores, r2_scores, rl_scores = [], [], []
    bleu_scores = []
    
    results_detailed = []
    
    print(f"\nEvaluating RAG on {len(GOLDEN_EVAL_SET)} golden test cases...")
    
    for idx, case in enumerate(GOLDEN_EVAL_SET):
        title = case["title"]
        context = case["context"]
        ref = case["reference"]
        
        # Override the retriever temporarily to use this exact context for the evaluation test
        # to isolate generator quality
        # Run process
        result = pipeline.process(title)
        
        # Override the context with our golden context so the generator gets the exact text
        rag_output = pipeline.explainer.explain_and_rewrite(title, context)
        gen_title = rag_output["rewritten_title"]
        explanation = rag_output["explanation"]
        
        # Tokenize words for metric calculation
        ref_tokens = ref.lower().split()
        gen_tokens = gen_title.lower().split()
        
        # Calculate BLEU (1-gram weight 0.5, 2-gram weight 0.5 for headlines)
        bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5), smoothing_function=chencherry.method1)
        bleu_scores.append(bleu)
        
        # Calculate ROUGE
        scores = scorer.score(ref, gen_title)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
        
        results_detailed.append({
            "id": idx + 1,
            "clickbait_title": title,
            "context": context,
            "human_reference": ref,
            "generated_title": gen_title,
            "explanation": explanation,
            "bleu": bleu,
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })
        
        print(f"\nCase {idx+1}:")
        print(f"  Input Title: {title}")
        print(f"  Generated:   {gen_title}")
        print(f"  Reference:   {ref}")
        print(f"  BLEU: {bleu:.4f} | ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        
    avg_bleu = np.mean(bleu_scores)
    avg_r1 = np.mean(r1_scores)
    avg_r2 = np.mean(r2_scores)
    avg_rl = np.mean(rl_scores)
    
    print("\n==========================================")
    print("RAG Generation Evaluation Results:")
    print(f"Average BLEU:    {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_r1:.4f}")
    print(f"Average ROUGE-2: {avg_r2:.4f}")
    print(f"Average ROUGE-L: {avg_rl:.4f}")
    print("==========================================")
    
    # Save evaluation report
    output_path = os.path.join(RESULTS_DIR, "rag_metrics.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RAG Generation Evaluation Report\n")
        f.write("==========================================\n")
        f.write(f"Average BLEU:    {avg_bleu:.4f}\n")
        f.write(f"Average ROUGE-1: {avg_r1:.4f}\n")
        f.write(f"Average ROUGE-2: {avg_r2:.4f}\n")
        f.write(f"Average ROUGE-L: {avg_rl:.4f}\n")
        f.write("==========================================\n\n")
        f.write("Detailed Results:\n")
        for res in results_detailed:
            f.write(f"ID: {res['id']}\n")
            f.write(f"Clickbait Title: {res['clickbait_title']}\n")
            f.write(f"Context: {res['context']}\n")
            f.write(f"Human Reference: {res['human_reference']}\n")
            f.write(f"Generated Title: {res['generated_title']}\n")
            f.write(f"Explanation: {res['explanation']}\n")
            f.write(f"Metrics: BLEU={res['bleu']:.4f}, ROUGE-1={res['rouge1']:.4f}, ROUGE-2={res['rouge2']:.4f}, ROUGE-L={res['rougeL']:.4f}\n")
            f.write("-" * 40 + "\n")
            
    print(f"Saved RAG evaluation metrics to {output_path}")

if __name__ == "__main__":
    evaluate_rag_pipeline()
