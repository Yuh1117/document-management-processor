import os

# Minimum cosine similarity (-1..1) between query and chunk vectors.
SEARCH_MIN_COSINE = 0.4
SEARCH_KNN_POOL_SIZE = 1000
SEARCH_RRF_K = 60
SEARCH_DEFAULT_CANDIDATE_SIZE = 100
SEARCH_CANDIDATE_MULTIPLIER = 10
SEARCH_MAX_CANDIDATE_SIZE = 1000

SNIPPET_MAX_CHARS = 320


QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")
FULL_TEXT_QUERY_FILE = os.path.join(QUERIES_DIR, "full_text.json")
SEMANTIC_QUERY_FILE = os.path.join(QUERIES_DIR, "semantic.json")
