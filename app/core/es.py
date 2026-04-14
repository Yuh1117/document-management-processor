from elasticsearch import Elasticsearch
from app.core.config import ELASTICSEARCH_HOST


class ElasticsearchClient:
    def __init__(self, host: str = ELASTICSEARCH_HOST) -> None:
        self.host = host
        self.client: Elasticsearch | None = None

    def get_client(self) -> Elasticsearch:
        if self.client is None:
            self.client = Elasticsearch(self.host)
        return self.client


es_client = ElasticsearchClient()
