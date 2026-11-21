import argparse
import json
import os
import sys
import tempfile
import mlflow
import mlflow.pyfunc
import ollama
from app.core.config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    OLLAMA_HOST,
    OLLAMA_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
)
from app.constants.defaults import (
    LLM_PROVIDER_OLLAMA,
    LLM_PROVIDER_OPENAI,
    OLLAMA_TIMEOUT,
)

MLFLOW_EXPERIMENT = "document-summarization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a new version of the summarizer in the MLflow Model Registry."
    )
    parser.add_argument(
        "--provider",
        choices=[LLM_PROVIDER_OLLAMA, LLM_PROVIDER_OPENAI],
        default=LLM_PROVIDER_OLLAMA,
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        help=(
            "Model to register, e.g. llama3.1:8b or gpt-4o-mini "
            f"(default: OLLAMA_MODEL_NAME={OLLAMA_MODEL_NAME} / "
            f"OPENAI_MODEL_NAME={OPENAI_MODEL_NAME})"
        ),
    )
    args = parser.parse_args()
    if not args.model:
        args.model = (
            OPENAI_MODEL_NAME
            if args.provider == LLM_PROVIDER_OPENAI
            else OLLAMA_MODEL_NAME
        )
    if not args.model:
        parser.error("--model is required (no default model configured)")
    return args


def normalize_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def ensure_model_pulled(model: str) -> None:
    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        pulled = {normalize_tag(m.model) for m in client.list().models if m.model}
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST} to verify the model: {e}")

    if normalize_tag(model) not in pulled:
        available = ", ".join(sorted(pulled)) or "none"
        sys.exit(
            f"Model '{model}' has not been pulled in Ollama "
            f"(available: {available}). "
            f"Run: docker exec dms-ollama ollama pull {model}"
        )


def ensure_openai_model(model: str) -> None:
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set")
    try:
        from app.utils.llm.openai_client import OpenAIClient

        OpenAIClient().client.models.retrieve(model)
    except Exception as e:
        sys.exit(f"Cannot verify OpenAI model '{model}': {e}")


def main():
    args = parse_args()
    provider, model = args.provider, args.model
    if provider == LLM_PROVIDER_OPENAI:
        ensure_openai_model(model)
    else:
        ensure_model_pulled(model)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    config = {"provider": provider, "model_name": model}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp)
        config_path = tmp.name

    try:
        with mlflow.start_run(run_name=f"register-{provider}-summarizer"):
            mlflow.log_params({"provider": provider, "model_name": model})
            if provider == LLM_PROVIDER_OLLAMA:
                mlflow.log_param("ollama_model", model)

            from app.utils.mlflow.summarizer import LLMSummarizer

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LLMSummarizer(),
                artifacts={"config": config_path},
                metadata={"provider": provider, "model_name": model},
            )
            run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)

        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name=MLFLOW_REGISTERED_MODEL_NAME,
            version=result.version,
            key="model_name",
            value=model,
        )
        client.set_model_version_tag(
            name=MLFLOW_REGISTERED_MODEL_NAME,
            version=result.version,
            key="provider",
            value=provider,
        )

        print(
            f"Registered: {MLFLOW_REGISTERED_MODEL_NAME} version {result.version} "
            f"(provider={provider}, model={model})"
        )

    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    main()
