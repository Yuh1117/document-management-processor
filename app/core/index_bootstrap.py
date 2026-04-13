import json
from elasticsearch import Elasticsearch
from app.core.config import DOCUMENTS_INDEX_MAPPING_FILE, ELASTICSEARCH_INDEX
from app.core.es import es_client


class IndexBootstrap:
    def __init__(
        self,
        es: Elasticsearch | None = None,
        index_name: str = ELASTICSEARCH_INDEX,
        mapping_file: str = DOCUMENTS_INDEX_MAPPING_FILE,
    ) -> None:
        self.es = es or es_client.get_client()
        self.index_name = index_name
        self.mapping_file = mapping_file

    def ensure_index_exists(self) -> None:
        try:
            if self.es.indices.exists(index=self.index_name):
                print(f"Index '{self.index_name}' already exists.")
                return

            with open(self.mapping_file, "r", encoding="utf-8") as f:
                body = json.load(f)

            self.es.indices.create(index=self.index_name, body=body)
            print(f"Successfully created index: {self.index_name}")

        except FileNotFoundError:
            print(f"Mapping file not found at {self.mapping_file}")
            raise
        except Exception as e:
            print(f"Failed to initialize Elasticsearch index: {e}")
            raise
