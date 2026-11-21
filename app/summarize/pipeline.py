import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.core.constants import DEFAULT_LANG
from app.indexing.chunking import chunk_text
from app.llm.base import LLMClient
from app.llm.constants import LLM_PROVIDER_OLLAMA, LLM_PROVIDER_OPENAI
from app.llm.factory import get_client
from app.summarize.cleaning import clean_text
from app.summarize.constants import SUMMARY_CHUNK_WORDS, SUMMARY_MAP_WORKERS
from app.summarize.prompts import (
    COMBINE_NOTE,
    EXCERPT_SEPARATOR,
    EXCERPTS_NOTE,
    PART_PROMPTS,
    SUMMARY_PROMPTS,
    pick,
)
from app.summarize.selection import select_representative

logger = logging.getLogger(__name__)


STRATEGY_SINGLE = "single"
STRATEGY_SELECT = "select"
STRATEGY_MAP_REDUCE = "map_reduce"

# How a document that does not fit in one call is handled, per provider:
# - local Ollama is slow, so keep one call and send representative excerpts;
# - OpenAI is fast and can run in parallel, so summarize every part (map-reduce).
LONG_DOCUMENT_STRATEGY = {
    LLM_PROVIDER_OLLAMA: STRATEGY_SELECT,
    LLM_PROVIDER_OPENAI: STRATEGY_MAP_REDUCE,
}


@dataclass
class SummaryResult:
    summary_text: str
    strategy: str
    input_chars: int
    cleaned_chars: int
    num_chunks: int = 1
    used_chunks: int = 1
    llm_calls: int = 1


def generate_summary(
    provider: str, model: str, text: str, language: str
) -> SummaryResult:
    client = get_client(provider)
    cleaned = clean_text(text)
    budget = client.max_input_chars

    if len(cleaned) <= budget:
        summary = summarize_once(client, model, cleaned, language)
        return SummaryResult(summary, STRATEGY_SINGLE, len(text), len(cleaned))

    chunks = chunk_text(cleaned, size=SUMMARY_CHUNK_WORDS, overlap=0)
    strategy = LONG_DOCUMENT_STRATEGY.get(provider, STRATEGY_SELECT)
    logger.info(
        "Long document (%d chars, %d chunks) -> strategy=%s",
        len(cleaned),
        len(chunks),
        strategy,
    )

    if strategy == STRATEGY_MAP_REDUCE:
        summary, calls = map_reduce(client, model, chunks, language)
        return SummaryResult(
            summary, strategy, len(text), len(cleaned), len(chunks), len(chunks), calls
        )

    summary, used = summarize_excerpts(client, model, chunks, language)
    return SummaryResult(summary, strategy, len(text), len(cleaned), len(chunks), used)


def summarize_once(
    client: LLMClient, model: str, text: str, language: str, note: str = ""
) -> str:
    template = pick(SUMMARY_PROMPTS, language, DEFAULT_LANG)
    return client.generate(model, note + template.format(text=text))


def summarize_excerpts(
    client: LLMClient, model: str, chunks: list[str], language: str
) -> tuple[str, int]:
    # Imported lazily: loading the embedding model is only needed here.
    from app.embedding.service import embedding_service

    vectors = embedding_service.encode_many(chunks)
    indexes = select_representative(
        chunks, vectors, client.max_input_chars, EXCERPT_SEPARATOR
    )
    excerpts = EXCERPT_SEPARATOR.join(chunks[i] for i in indexes)
    note = pick(EXCERPTS_NOTE, language, DEFAULT_LANG)
    return summarize_once(client, model, excerpts, language, note), len(indexes)


def group_chunks(chunks: list[str], max_chars: int) -> list[str]:
    """Merge consecutive chunks into parts of at most max_chars."""
    parts: list[str] = []
    current = ""
    for chunk in chunks:
        if current and len(current) + 1 + len(chunk) > max_chars:
            parts.append(current)
            current = chunk
        else:
            current = f"{current}\n{chunk}" if current else chunk
    if current:
        parts.append(current)
    return parts


def map_reduce(
    client: LLMClient, model: str, chunks: list[str], language: str
) -> tuple[str, int]:
    parts = group_chunks(chunks, client.max_input_chars)
    template = pick(PART_PROMPTS, language, DEFAULT_LANG)

    def summarize_part(item: tuple[int, str]) -> str:
        index, part = item
        prompt = template.format(index=index, total=len(parts), text=part)
        return client.generate(model, prompt)

    with ThreadPoolExecutor(max_workers=SUMMARY_MAP_WORKERS) as pool:
        partials = list(pool.map(summarize_part, enumerate(parts, start=1)))
    calls = len(parts)

    combined = "\n\n".join(partials)
    if len(combined) > client.max_input_chars and len(parts) > 1:
        # Still too long: reduce the partial summaries once more.
        summary, extra = map_reduce(
            client,
            model,
            chunk_text(combined, size=SUMMARY_CHUNK_WORDS, overlap=0),
            language,
        )
        return summary, calls + extra

    note = pick(COMBINE_NOTE, language, DEFAULT_LANG)
    combined = combined[: client.max_input_chars]
    return summarize_once(client, model, combined, language, note), calls + 1
