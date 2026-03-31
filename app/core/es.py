from elasticsearch import Elasticsearch
from app.core.config import ELASTICSEARCH_HOST


class ElasticsearchClient:
    def __init__(self, host: str = ELASTICSEARCH_HOST) -> None:
        self._host = host
        self._client: Elasticsearch | None = None

    def get_client(self) -> Elasticsearch:
        if self._client is None:
            self._client = Elasticsearch(self._host)
        return self._client


es_client = ElasticsearchClient()
