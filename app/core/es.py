from elasticsearch import Elasticsearch
from app.core.config import ELASTICSEARCH_HOST

_es_client: Elasticsearch | None = None


def get_es() -> Elasticsearch:
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch(ELASTICSEARCH_HOST)
    return _es_client
