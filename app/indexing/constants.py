import os

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


DOCUMENTS_INDEX_MAPPING_FILE = os.path.join(
    os.path.dirname(__file__), "mappings", "documents_index.json"
)
