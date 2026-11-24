OLLAMA_TIMEOUT = 300.0
# ~4-5k tokens, leaving room in OLLAMA_NUM_CTX for the prompt and the answer.
OLLAMA_MAX_INPUT_CHARS = 12000
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 768
OLLAMA_KEEP_ALIVE = "30m"

LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_OPENAI = "openai"

OPENAI_TIMEOUT = 120.0
OPENAI_MAX_INPUT_CHARS = 60000
OPENAI_MAX_OUTPUT_TOKENS = 1024
# Chat answers: the Responses API counts reasoning tokens in this limit too.
OPENAI_CHAT_MAX_OUTPUT_TOKENS = 2048
# Reasoning effort for chat answers; dropped automatically for non-reasoning models.
OPENAI_REASONING_EFFORT = "low"

# Retries before the first streamed piece for providers without built-in retries.
LLM_RETRY_ATTEMPTS = 2
LLM_RETRY_BASE_DELAY = 1.0
