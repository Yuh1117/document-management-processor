import os
import tempfile


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


RABBITMQ_URL = os.getenv("RABBITMQ_URL")
RABBITMQ_DOCUMENT_QUEUE = os.getenv("RABBITMQ_DOCUMENT_PROCESS_QUEUE")
RABBITMQ_PROCESSING_RESULT_QUEUE = os.getenv("RABBITMQ_DOCUMENT_RESULT_QUEUE")

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONSTANTS_DIR = os.path.join(BASE_DIR, "constants")
ES_MAPPINGS_DIR = os.path.join(CONSTANTS_DIR, "es_mappings")
ES_QUERIES_DIR = os.path.join(CONSTANTS_DIR, "es_queries")
DOCUMENTS_INDEX_MAPPING_FILE = os.path.join(ES_MAPPINGS_DIR, "documents_index.json")
FULL_TEXT_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "full_text.json")
SEMANTIC_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "semantic.json")
HYBRID_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "hybrid.json")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

AWS_S3_ACCESS_KEY = os.getenv("AWS_S3_ACCESS_KEY")
AWS_S3_SECRET_KEY = os.getenv("AWS_S3_SECRET_KEY")
AWS_S3_REGION = os.getenv("AWS_S3_REGION")

# Worker: temp storage and OCR
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())
LAPLACIAN_VAR_THRESHOLD = float(os.getenv("LAPLACIAN_VAR_THRESHOLD", "100.0"))
OCR_USE_GPU = _env_bool("OCR_USE_GPU", False)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Data validation
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "150"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "150"))
MIN_CONTRAST_THRESHOLD = float(os.getenv("MIN_CONTRAST_THRESHOLD", "15.0"))
VALIDATE_ALL_PDF_PAGES = _env_bool("VALIDATE_ALL_PDF_PAGES", True)

# OCR monitoring
OCR_MIN_QUALITY_SCORE = int(os.getenv("OCR_MIN_QUALITY_SCORE", "60"))
OCR_MAX_INVALID_CHAR_RATIO = float(os.getenv("OCR_MAX_INVALID_CHAR_RATIO", "0.3"))

SENTENCE_TRANSFORMER_MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
SUMMARIZE_PROMPT_VERSION = os.getenv("SUMMARIZE_PROMPT_VERSION")

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
MLFLOW_SUMMARIZE_MODEL_URI = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@champion"
