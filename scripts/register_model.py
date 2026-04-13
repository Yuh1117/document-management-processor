import json
import os
import tempfile
import mlflow
import mlflow.pyfunc
from app.core.config import (
    MLFLOW_TRACKING_URI,
    GEMINI_MODEL_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
)

MLFLOW_EXPERIMENT = "document-summarization"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

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
                metadata={"model_name": GEMINI_MODEL_NAME},
            )
            run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)

        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name=MLFLOW_REGISTERED_MODEL_NAME,
            version=result.version,
            key="model_name",
            value=GEMINI_MODEL_NAME,
        )

        print(
            f"Registered: {MLFLOW_REGISTERED_MODEL_NAME} version {result.version} "
            f"(model={GEMINI_MODEL_NAME})"
        )

    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    main()
