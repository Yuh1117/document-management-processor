import logging

import mlflow
from google import genai

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    SUMMARIZE_PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

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

AVAILABLE_MODELS = [
    {
        "id": GEMINI_MODEL_NAME,
        "provider": "gemini",
        "is_default": True,
        "prompt_version": SUMMARIZE_PROMPT_VERSION or "v1",
    },
]

DEFAULT_LANG = "vi"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class SummarizeService:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def list_models(self) -> list[dict]:
        return AVAILABLE_MODELS

    def summarize(
        self, text: str, language: str = DEFAULT_LANG, model_id: str | None = None
    ) -> dict:
        model_name = model_id or GEMINI_MODEL_NAME
        prompt_version = SUMMARIZE_PROMPT_VERSION or "v1"

        template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
        prompt = template.format(text=text)

        logger.info("Summarizing with model=%s", model_name)
        response = self._client.models.generate_content(
            model=model_name, contents=prompt
        )

        summary_text = response.text
        self._log_to_mlflow(
            model_name, prompt_version, language, len(text), len(summary_text)
        )

        return {
            "summary_text": summary_text,
            "model_version": model_name,
            "prompt_version": prompt_version,
        }

    @staticmethod
    def _log_to_mlflow(
        model_version: str,
        prompt_version: str,
        language: str,
        input_length: int,
        output_length: int,
    ) -> None:
        try:
            mlflow.set_experiment("document-summarization")
            with mlflow.start_run():
                mlflow.log_params(
                    {
                        "model_version": model_version,
                        "prompt_version": prompt_version,
                        "language": language,
                    }
                )
                mlflow.log_metrics(
                    {
                        "input_length": input_length,
                        "output_length": output_length,
                        "compression_ratio": round(output_length / input_length, 4)
                        if input_length
                        else 0,
                    }
                )
        except Exception as e:
            logger.warning("MLflow logging failed (non-fatal): %s", e)


summarize_service = SummarizeService()
