import os
import tempfile

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
RABBITMQ_DOCUMENT_QUEUE = os.getenv("RABBITMQ_DOCUMENT_QUEUE")

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

AWS_S3_ACCESS_KEY = os.getenv("AWS_S3_ACCESS_KEY")
AWS_S3_SECRET_KEY = os.getenv("AWS_S3_SECRET_KEY")
AWS_S3_REGION = os.getenv("AWS_S3_REGION")

# Worker: temp storage and OCR
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())
LAPLACIAN_VAR_THRESHOLD = float(os.getenv("LAPLACIAN_VAR_THRESHOLD", "100.0"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SENTENCE_TRANSFORMER_MODEL_NAME = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONSTANTS_DIR = os.path.join(BASE_DIR, "constants")
ES_MAPPINGS_DIR = os.path.join(CONSTANTS_DIR, "es_mappings")
ES_QUERIES_DIR = os.path.join(CONSTANTS_DIR, "es_queries")
DOCUMENTS_INDEX_MAPPING_FILE = os.path.join(ES_MAPPINGS_DIR, "documents_index.json")
FULL_TEXT_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "full_text.json")
SEMANTIC_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "semantic.json")
HYBRID_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "hybrid.json")
