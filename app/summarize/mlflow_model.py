import json
import logging
from dataclasses import asdict

import mlflow.pyfunc
import pandas as pd

from app.core.constants import DEFAULT_LANG
from app.llm.constants import LLM_PROVIDER_OLLAMA
from app.summarize.pipeline import generate_summary

logger = logging.getLogger(__name__)


class LLMSummarizer(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping an LLM (Ollama or OpenAI) for summarization.

    Only the provider and model name are stored in the artifact; credentials
    such as OPENAI_API_KEY are read from the environment at runtime.
    predict() returns a dict (see SummaryResult).
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

    def predict(self, context, model_input: pd.DataFrame) -> dict:
        if hasattr(model_input, "to_dict"):
            row = model_input.iloc[0].to_dict()
        elif isinstance(model_input, dict):
            row = model_input
        else:
            row = dict(model_input)

        text = row["text"]
        language = row.get("language", DEFAULT_LANG)

        return asdict(generate_summary(self.provider, self.model_name, text, language))
