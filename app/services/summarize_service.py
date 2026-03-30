import google.generativeai as genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, SUMMARIZE_PROMPT_VERSION

SUMMARIZE_PROMPT_TEMPLATE = """Bạn là trợ lý AI chuyên tóm tắt tài liệu. Hãy tóm tắt nội dung sau một cách ngắn gọn, rõ ràng và đầy đủ các ý chính.

Yêu cầu:
- Tóm tắt bằng ngôn ngữ của văn bản gốc
- Giữ lại các thông tin quan trọng, số liệu, tên riêng
- Trình bày dưới dạng đoạn văn mạch lạc
- Độ dài tóm tắt khoảng 15-25% nội dung gốc

Nội dung cần tóm tắt:
---
{text}
---

Tóm tắt:"""


class SummarizeService:
    def __init__(self) -> None:
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def summarize(self, text: str) -> dict:
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(text=text)
        response = self._model.generate_content(prompt)
        return {
            "summary_text": response.text,
            "model_version": GEMINI_MODEL_NAME,
            "prompt_version": SUMMARIZE_PROMPT_VERSION,
        }


summarize_service = SummarizeService()
