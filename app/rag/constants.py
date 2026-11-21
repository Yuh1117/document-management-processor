import os

RAG_TEMPERATURE = 0.1
RAG_RETRIEVE_SIZE = 10
RAG_TOP_CHUNKS = 5
RAG_WINDOW_WORDS = 150
RAG_MAX_PASSAGES = 5
# Minimum cosine similarity (-1..1) between the question and a passage.
RAG_MIN_COSINE = 0.8
RAG_SNIPPET_CHARS = 280
RAG_MAX_QUESTION_CHARS = 500


QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")
RAG_FULL_TEXT_QUERY_FILE = os.path.join(QUERIES_DIR, "full_text.json")
RAG_SEMANTIC_QUERY_FILE = os.path.join(QUERIES_DIR, "semantic.json")
