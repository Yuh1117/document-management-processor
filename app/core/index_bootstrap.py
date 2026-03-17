import json
from app.core.config import DOCUMENTS_INDEX_MAPPING_FILE, ELASTICSEARCH_INDEX
from app.core.es import get_es


def ensure_documents_index_exists() -> None:
    es = get_es()

    try:
        if es.indices.exists(index=ELASTICSEARCH_INDEX):
            print(f"Index '{ELASTICSEARCH_INDEX}' already exists.")
            return

        with open(DOCUMENTS_INDEX_MAPPING_FILE, "r", encoding="utf-8") as f:
            body = json.load(f)

        es.indices.create(index=ELASTICSEARCH_INDEX, body=body)
        print(f"Successfully created index: {ELASTICSEARCH_INDEX}")

    except FileNotFoundError:
        print(f"Mapping file not found at {DOCUMENTS_INDEX_MAPPING_FILE}")
        raise
    except Exception as e:
        print(f"Failed to initialize Elasticsearch index: {e}")
        raise
