import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from google import genai
from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MLFLOW_SUMMARIZE_MODEL_URI,
    MLFLOW_TRACKING_URI,
    SUMMARIZE_PROMPT_VERSION,
)
from app.utils.mlflow.gemini_summarizer import PROMPTS

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = [
    {
        "id": GEMINI_MODEL_NAME,
        "provider": "gemini",
        "is_default": True,
        "prompt_version": SUMMARIZE_PROMPT_VERSION,
    },
]

DEFAULT_LANG = "vi"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class SummarizeService:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._mlflow_model = self._load_mlflow_model()

    def list_models(self) -> list[dict]:
        return AVAILABLE_MODELS

    def summarize(self, text: str, language: str = DEFAULT_LANG) -> dict:
        model_name = GEMINI_MODEL_NAME
        prompt_version = SUMMARIZE_PROMPT_VERSION

        if self._mlflow_model is not None:
            summary_text = self._predict_via_mlflow(text, language)
        else:
            summary_text = self._predict_direct(text, language, model_name)

        self._log_to_mlflow(
            model_name, prompt_version, language, len(text), len(summary_text)
        )

        return {
            "summary_text": summary_text,
            "model_name": model_name,
            "prompt_version": prompt_version,
        }

    def _load_mlflow_model(self):
        try:
            model = mlflow.pyfunc.load_model(MLFLOW_SUMMARIZE_MODEL_URI)
            logger.info("Loaded MLflow model from %s", MLFLOW_SUMMARIZE_MODEL_URI)
            return model
        except Exception as e:
            logger.warning(
                "Could not load MLflow model (%s), using direct Gemini call: %s",
                MLFLOW_SUMMARIZE_MODEL_URI,
                e,
            )
            return None

    def _predict_via_mlflow(self, text: str, language: str) -> str:
        input_df = pd.DataFrame([{"text": text, "language": language}])
        return self._mlflow_model.predict(input_df)

    def _predict_direct(self, text: str, language: str, model_name: str) -> str:
        template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
        prompt = template.format(text=text)

        logger.info("Summarizing (direct) with model=%s", model_name)
        response = self._client.models.generate_content(
            model=model_name, contents=prompt
        )
        return response.text

    @staticmethod
    def _log_to_mlflow(
        model_name: str,
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
                        "model_name": model_name,
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
