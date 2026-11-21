import json
import logging
import mlflow.pyfunc
import pandas as pd
from app.constants.defaults import DEFAULT_LANG, LLM_PROVIDER_OLLAMA
from app.utils.llm import get_client

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


def generate_summary(provider: str, model: str, text: str, language: str) -> str:
    client = get_client(provider)
    if len(text) > client.max_input_chars:
        logger.warning(
            "Input truncated from %d to %d chars for summarization",
            len(text),
            client.max_input_chars,
        )
        text = text[: client.max_input_chars]

    template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
    return client.generate(model, template.format(text=text))


class LLMSummarizer(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping an LLM (Ollama or OpenAI) for summarization.

    Only the provider and model name are stored in the artifact; credentials
    such as OPENAI_API_KEY are read from the environment at runtime.
    """

    def load_context(self, context):
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self.provider = config.get("provider", LLM_PROVIDER_OLLAMA)
        logger.info(
            "%s loaded: provider=%s model=%s",
            type(self).__name__,
            self.provider,
            self.model_name,
        )

    def predict(self, context, model_input: pd.DataFrame) -> str:
        if hasattr(model_input, "to_dict"):
            row = model_input.iloc[0].to_dict()
        elif isinstance(model_input, dict):
            row = model_input
        else:
            row = dict(model_input)

        text = row["text"]
        language = row.get("language", DEFAULT_LANG)

        return generate_summary(self.provider, self.model_name, text, language)

