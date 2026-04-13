from datetime import datetime
import logging
import threading

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import HTTPException
from google import genai

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_SUMMARIZE_MODEL_URI,
    MLFLOW_TRACKING_URI,
    SUMMARIZE_PROMPT_VERSION,
)
from app.utils.mlflow.gemini_summarizer import PROMPTS

logger = logging.getLogger(__name__)

DEFAULT_LANG = "vi"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class SummarizeService:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.mlflow_model = self.load_mlflow_model()
        self.prompt_version = SUMMARIZE_PROMPT_VERSION
        self.lock = threading.Lock()

    def list_models(self) -> list:
        client = mlflow.MlflowClient()
        try:
            versions = client.search_model_versions(
                f"name='{MLFLOW_REGISTERED_MODEL_NAME}'"
            )
        except Exception as e:
            logger.warning("Failed to list MLflow model versions: %s", e)
            versions = []

        active_version = None
        if self.mlflow_model is not None:
            try:
                active_version = self.mlflow_model.metadata.model_uuid
            except Exception:
                pass

        models = []
        for mv in versions:
            metadata = {}
            try:
                run = client.get_run(mv.run_id)
                metadata = run.data.params
            except Exception:
                pass

            is_active = active_version is not None and mv.run_id == getattr(
                self.mlflow_model.metadata, "run_id", None
            )

            models.append(
                {
                    "version": mv.version,
                    "model_name": metadata.get("gemini_model"),
                    "is_active": is_active,
                    "created_at": datetime.fromtimestamp(
                        mv.creation_timestamp / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return models

    def reload_model(self) -> dict:
        with self.lock:
            old_model_name = self.model_name
            self.mlflow_model = self.load_mlflow_model()
            logger.info("Model reloaded: %s -> %s", old_model_name, self.model_name)
            return {
                "previous_model": old_model_name,
                "current_model": self.model_name,
                "mlflow_model_uri": MLFLOW_SUMMARIZE_MODEL_URI,
                "status": "loaded" if self.mlflow_model is not None else "fallback",
            }

    def summarize(self, text: str, language: str = DEFAULT_LANG) -> dict:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        try:
            if self.mlflow_model is not None:
                summary_text = self.predict_via_mlflow(text, language)
            else:
                summary_text = self.predict_direct(text, language)
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

        self.log_to_mlflow(
            self.model_name,
            self.prompt_version,
            language,
            len(text),
            len(summary_text),
        )

        return {
            "summary_text": summary_text,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
        }

    def load_mlflow_model(self):
        try:
            model = mlflow.pyfunc.load_model(MLFLOW_SUMMARIZE_MODEL_URI)
            metadata = model.metadata.metadata or {}
            self.model_name = metadata.get("model_name")
            logger.info("Loaded MLflow model: %s (prompt %s)", self.model_name)
            return model
        except Exception as e:
            self.model_name = GEMINI_MODEL_NAME
            logger.warning(
                "Could not load MLflow model (%s), falling back to config: %s",
                MLFLOW_SUMMARIZE_MODEL_URI,
                e,
            )
            return None

    def predict_via_mlflow(self, text: str, language: str) -> str:
        input_df = pd.DataFrame([{"text": text, "language": language}])
        return self.mlflow_model.predict(input_df)

    def predict_direct(self, text: str, language: str) -> str:
        template = PROMPTS.get(language, PROMPTS[DEFAULT_LANG])
        prompt = template.format(text=text)

        logger.info("Summarizing (direct) with model=%s", self.model_name)
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text

    @staticmethod
    def log_to_mlflow(
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
