import json
import os
import tempfile
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from app.core.config import MLFLOW_TRACKING_URI, GEMINI_MODEL_NAME

MODEL_REGISTRY_NAME = "summarization_model"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("document-summarization")

    config = {"model_name": GEMINI_MODEL_NAME}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp)
        config_path = tmp.name

    try:
        with mlflow.start_run(run_name="register-gemini-summarizer"):
            mlflow.log_param("gemini_model", GEMINI_MODEL_NAME)

            from app.utils.mlflow.gemini_summarizer import GeminiSummarizer

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=GeminiSummarizer(),
                artifacts={"config": config_path},
            )
            run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
        print(f"Registered: {MODEL_REGISTRY_NAME} version {result.version}")

        client = MlflowClient()
        client.set_registered_model_alias(
            MODEL_REGISTRY_NAME, "champion", result.version
        )
        print(f"Alias 'champion' set to version {result.version}")

    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    main()
