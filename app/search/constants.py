import os

# Minimum cosine similarity (-1..1) between query and chunk vectors.
# multilingual-e5 squeezes cosines into ~0.7-0.9: unrelated chunks still score
# 0.70-0.79, relevant ones 0.81+. Re-calibrate if the embedding model changes.
SEARCH_MIN_COSINE = 0.80
SEARCH_KNN_POOL_SIZE = 1000
SEARCH_RRF_K = 60
# Documents taken from each retriever before RRF; fixed so paging stays stable.
SEARCH_HYBRID_WINDOW = 100
SEARCH_DEFAULT_CANDIDATE_SIZE = 100
SEARCH_CANDIDATE_MULTIPLIER = 10
SEARCH_MAX_CANDIDATE_SIZE = 1000

SNIPPET_MAX_CHARS = 320


QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")
FULL_TEXT_QUERY_FILE = os.path.join(QUERIES_DIR, "full_text.json")
SEMANTIC_QUERY_FILE = os.path.join(QUERIES_DIR, "semantic.json")
