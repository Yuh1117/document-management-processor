import json
import logging
import mlflow.pyfunc
import pandas as pd
from google import genai
from app.core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

PROMPTS = {
    "vi": (
        "Bạn là trợ lý AI chuyên tóm tắt tài liệu. Hãy tóm tắt nội dung sau "
        "một cách ngắn gọn, rõ ràng và đầy đủ các ý chính.\n\n"
        "Yêu cầu:\n"
        "- Tóm tắt bằng tiếng Việt\n"
        "- Giữ lại các thông tin quan trọng, số liệu, tên riêng\n"
        "- Trình bày dưới dạng đoạn văn mạch lạc\n"
        "- Độ dài tóm tắt khoảng 15-25% nội dung gốc\n\n"
        "Nội dung cần tóm tắt:\n---\n{text}\n---\n\nTóm tắt:"
    ),
    "en": (
        "You are an AI assistant specialized in document summarization. "
        "Summarize the following content concisely, clearly, and covering all key points.\n\n"
        "Requirements:\n"
        "- Summarize in English\n"
        "- Retain important information, figures, and proper nouns\n"
        "- Present as coherent paragraphs\n"
        "- Summary length should be about 15-25% of the original\n\n"
        "Content to summarize:\n---\n{text}\n---\n\nSummary:"
    ),
}

DEFAULT_LANG = "vi"


class GeminiSummarizer(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping Google Gemini for summarization."""

    def load_context(self, context):
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("GeminiSummarizer loaded: model=%s", self.model_name)

    def predict(self, context, model_input: pd.DataFrame) -> str:
        if hasattr(model_input, "to_dict"):
            row = model_input.iloc[0].to_dict()
        elif isinstance(model_input, dict):
            row = model_input
        else:
            row = dict(model_input)

        text = row["text"]
        language = row.get("language", DEFAULT_LANG)

        template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
        prompt = template.format(text=text)

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text
