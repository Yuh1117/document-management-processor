import os

CHUNK_SIZE = 320
CHUNK_OVERLAP = 64


DOCUMENTS_INDEX_MAPPING_FILE = os.path.join(
    os.path.dirname(__file__), "mappings", "documents_index.json"
)
