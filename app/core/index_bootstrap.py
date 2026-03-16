import json
from app.core.config import DOCUMENTS_INDEX_MAPPING_FILE, ELASTICSEARCH_INDEX
from app.core.es import get_es


def ensure_documents_index_exists() -> None:
    es = get_es()

    if es.indices.exists(index=ELASTICSEARCH_INDEX):
        return

    with open(DOCUMENTS_INDEX_MAPPING_FILE, "r", encoding="utf-8") as f:
        body = json.load(f)

    es.indices.create(index=ELASTICSEARCH_INDEX, body=body)
