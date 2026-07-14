"""
Generator module for RAG pipeline.
Uses an LLM (via OpenAI API) to explain why a title is clickbait 
and write an objective alternative headline. If no API key is found, 
falls back to a rule-based generator.
"""

import os
import json
import re

class ClickbaitExplainer:
    """
    RAG Generator that uses OpenAI GPT-4o-mini or a rule-based fallback 
    to explain and rewrite clickbait headlines.
    """
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            print("OpenAI API Key detected. Using GPT-4o-mini for generation.")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI package not installed. Falling back to rule-based generation.")
                self.client = None
        else:
            print("No OpenAI API Key found in environment. Using rule-based fallback generation.")
            self.client = None

    def explain_and_rewrite(self, title, context):
        """
        Generates an explanation and alternative headline.
        Returns a dictionary with keys: 'explanation' and 'rewritten_title'.
        """
        if self.client:
            try:
                return self._call_openai(title, context)
            except Exception as e:
                print(f"OpenAI API error: {e}. Falling back to rule-based generation.")
                
        return self._fallback_generation(title, context)

    def _call_openai(self, title, context):
        prompt = f"""Bạn là một chuyên gia kiểm chứng thông tin báo chí Việt Nam. 
Nhiệm vụ của bạn là phân tích một tiêu đề bị nghi ngờ là giật tít (clickbait), đối chiếu nó với phần tóm tắt/mở đầu của bài báo (context), sau đó:
1. Giải thích chi tiết vì sao tiêu đề này bị xem là clickbait (ví dụ: phóng đại cảm xúc, giấu thông tin quan trọng, đặt câu hỏi tu từ kích thích tò mò, gây hiểu lầm, v.v.).
2. Đề xuất một tiêu đề mới khách quan, trung thực, tóm tắt chính xác nội dung bài viết và loại bỏ mọi yếu tố clickbait.

Dữ liệu đầu vào:
- Tiêu đề clickbait: "{title}"
- Nội dung mở đầu bài viết (Context): "{context}"

Hãy trả về kết quả dưới định dạng JSON duy nhất, có cấu trúc như sau:
{{
  "explanation": "Lời giải thích bằng tiếng Việt...",
  "rewritten_title": "Tiêu đề đề xuất bằng tiếng Việt..."
}}
Chú ý: Chỉ trả về chuỗi JSON hợp lệ, không kèm theo markdown block hay văn bản thừa nào khác.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON objects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            # Clean text if markdown formatting is returned
            cleaned_text = re.sub(r"```json\s*|\s*```", "", result_text).strip()
            return json.loads(cleaned_text)

    def _fallback_generation(self, title, context):
        """
        Rule-based generator when LLM is unavailable.
        Uses structural features of the clickbait title and summarizes the context.
        """
        # Determine clickbait type based on patterns
        reasons = []
        if '?' in title:
            reasons.append("sử dụng câu hỏi tu từ kích thích sự tò mò của độc giả thay vì cung cấp thông tin trực tiếp")
        if '!' in title:
            reasons.append("sử dụng dấu chấm than để cường điệu hóa hoặc gây chú ý quá mức")
        
        vague_matches = [w for w in ["người này", "bí mật ấy", "sự thật về", "điều đó", "nơi này", "ai đó"] if w in title.lower()]
        if vague_matches:
            reasons.append(f"sử dụng các cụm từ mơ hồ như '{vague_matches[0]}' để cố tình che giấu thông tin quan trọng")
            
        exag_matches = [w for w in ["không thể tin nổi", "ngỡ ngàng", "gây sốc", "chấn động", "kinh hoàng", "bất ngờ"] if w in title.lower()]
        if exag_matches:
            reasons.append(f"sử dụng các từ ngữ phóng đại cảm xúc mạnh như '{exag_matches[0]}' nhằm lôi kéo tương tác")
            
        if not reasons:
            reasons.append("tiêu đề có cấu trúc giật tít, phóng đại hoặc mập mờ so với nội dung thực tế của bài báo")

        explanation = (
            f"Tiêu đề bị đánh giá là clickbait vì đã {', '.join(reasons)}. "
            f"Đối chiếu với nội dung bài báo, tiêu đề đã cố tình tạo khoảng cách thông tin (curiosity gap) để kích thích người dùng bấm vào xem."
        )
        
        # Rewrite the title based on the context's first sentence or key parts
        rewritten_title = title
        if context and context != "Nội dung bài báo đang được cập nhật.":
            # Extract first sentence of the context
            sentences = re.split(r'[.\n]', context)
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 15:
                rewritten_title = first_sentence
                # Clean up punctuation and shorten if too long
                if rewritten_title.endswith(','):
                    rewritten_title = rewritten_title[:-1]
                if len(rewritten_title) > 90:
                    words = rewritten_title.split()
                    rewritten_title = " ".join(words[:15]) + "..."
            else:
                rewritten_title = "Cập nhật thông tin chi tiết về bài báo liên quan đến: " + title.replace("?", "").replace("!", "")
        else:
            # Simple clean up of clickbait words
            clean_title = title
            for w in vague_matches + exag_matches:
                clean_title = re.sub(w, "", clean_title, flags=re.IGNORECASE)
            rewritten_title = clean_title.replace("?", "").replace("!", "").strip()
            
        return {
            "explanation": explanation,
            "rewritten_title": rewritten_title
        }

if __name__ == "__main__":
    explainer = ClickbaitExplainer()
    
    # Test fallback
    test_title = "Không thể tin nổi: Sự thật về người này đã bị phanh phui!"
    test_context = "Công an thành phố Hà Nội vừa tiến hành bắt giữ đối tượng Nguyễn Văn A để điều tra về hành vi lừa đảo chiếm đoạt tài sản công dân."
    
    result = explainer.explain_and_rewrite(test_title, test_context)
    print(f"Clickbait Title: {test_title}")
    print(f"Context: {test_context}")
    print(f"Explanation:\n{result['explanation']}")
    print(f"Rewritten Title:\n{result['rewritten_title']}")
