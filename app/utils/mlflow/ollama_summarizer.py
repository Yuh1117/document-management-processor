import json
import logging
import mlflow.pyfunc
import pandas as pd
import ollama
from app.constants.defaults import DEFAULT_LANG
from app.core.config import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_INPUT_CHARS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)

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


def generate_summary(
    client: ollama.Client, model: str, text: str, language: str
) -> str:
    if len(text) > OLLAMA_MAX_INPUT_CHARS:
        logger.warning(
            "Input truncated from %d to %d chars for summarization",
            len(text),
            OLLAMA_MAX_INPUT_CHARS,
        )
        text = text[:OLLAMA_MAX_INPUT_CHARS]

    template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
    response = client.generate(
        model=model,
        prompt=template.format(text=text),
        options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    return response.response


class OllamaSummarizer(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping a local Ollama model for summarization."""

    def load_context(self, context):
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self.client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        logger.info("OllamaSummarizer loaded: model=%s", self.model_name)

    def predict(self, context, model_input: pd.DataFrame) -> str:
        if hasattr(model_input, "to_dict"):
            row = model_input.iloc[0].to_dict()
        elif isinstance(model_input, dict):
            row = model_input
        else:
            row = dict(model_input)

        text = row["text"]
        language = row.get("language", DEFAULT_LANG)

        return generate_summary(self.client, self.model_name, text, language)
