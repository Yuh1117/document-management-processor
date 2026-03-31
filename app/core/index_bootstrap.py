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
        self._es = es or es_client.get_client()
        self._index_name = index_name
        self._mapping_file = mapping_file

    def ensure_index_exists(self) -> None:
        try:
            if self._es.indices.exists(index=self._index_name):
                print(f"Index '{self._index_name}' already exists.")
                return

            with open(self._mapping_file, "r", encoding="utf-8") as f:
                body = json.load(f)

            self._es.indices.create(index=self._index_name, body=body)
            print(f"Successfully created index: {self._index_name}")

        except FileNotFoundError:
            print(f"Mapping file not found at {self._mapping_file}")
            raise
        except Exception as e:
            print(f"Failed to initialize Elasticsearch index: {e}")
            raise
