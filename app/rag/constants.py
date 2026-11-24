import os

RAG_TEMPERATURE = 0.1
RAG_RETRIEVE_SIZE = 20
RAG_TOP_CHUNKS = 8
RAG_WINDOW_WORDS = 250
RAG_MAX_PASSAGES = 8
# Minimum cosine similarity (-1..1) between the question and a passage.
RAG_MIN_COSINE = 0.8
RAG_SNIPPET_CHARS = 280
RAG_MAX_QUESTION_CHARS = 500


QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")
RAG_FULL_TEXT_QUERY_FILE = os.path.join(QUERIES_DIR, "full_text.json")
RAG_SEMANTIC_QUERY_FILE = os.path.join(QUERIES_DIR, "semantic.json")

# Share of the model's input budget that history and a read document may use.
RAG_HISTORY_INPUT_SHARE = 0.2
RAG_READ_INPUT_SHARE = 0.4

# Tool calling, per answer: model rounds, tool calls in total, and whole-document
# reads (the costliest tool) among them.
RAG_MAX_ROUNDS = 5
RAG_MAX_TOOL_CALLS = 6
RAG_MAX_READS = 2
RAG_FIND_LIMIT = 8
RAG_READ_MAX_CHUNKS = 400
