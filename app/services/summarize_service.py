from google import genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, SUMMARIZE_PROMPT_VERSION

PROMPTS = {
    "vi": """Bạn là trợ lý AI chuyên tóm tắt tài liệu. Hãy tóm tắt nội dung sau một cách ngắn gọn, rõ ràng và đầy đủ các ý chính.

Yêu cầu:
- Tóm tắt bằng tiếng Việt
- Giữ lại các thông tin quan trọng, số liệu, tên riêng
- Trình bày dưới dạng đoạn văn mạch lạc
- Độ dài tóm tắt khoảng 15-25% nội dung gốc

Nội dung cần tóm tắt:
---
{text}
---

Tóm tắt:""",
    "en": """You are an AI assistant specialized in document summarization. Summarize the following content concisely, clearly, and covering all key points.

Requirements:
- Summarize in English
- Retain important information, figures, and proper nouns
- Present as coherent paragraphs
- Summary length should be about 15-25% of the original

Content to summarize:
---
{text}
---

Summary:""",
}

DEFAULT_LANG = "vi"


class SummarizeService:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def summarize(self, text: str, language: str = DEFAULT_LANG) -> dict:
        template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
        prompt = template.format(text=text)
        response = self._client.models.generate_content(
            model=GEMINI_MODEL_NAME, contents=prompt
        )
        return {
            "summary_text": response.text,
            "model_version": GEMINI_MODEL_NAME,
            "prompt_version": SUMMARIZE_PROMPT_VERSION,
        }


summarize_service = SummarizeService()
