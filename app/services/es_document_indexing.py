"""Chunking + Elasticsearch document indexing shared by the OCR worker and HTTP reindex."""

import logging
import time
from typing import Any

from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ELASTICSEARCH_INDEX,
    ES_INDEX_MAX_RETRIES,
    ES_INDEX_RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunks.append(" ".join(chunk_words))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def build_chunk_document_body(
    doc_id: int,
    chunk_index: int,
    chunk: str,
    content_vector: list[float],
    owner_id: Any,
    folder_id: Any,
    doc_name: str | None,
) -> dict:
    doc_body: dict = {
        "chunk_id": f"{doc_id}_{chunk_index}",
        "document_id": str(doc_id),
        "owner_id": str(owner_id) if owner_id is not None else None,
        "folder_id": str(folder_id) if folder_id is not None else None,
        "content": chunk,
        "content_vector": content_vector,
    }
    if doc_name is not None:
        doc_body["name"] = doc_name
    return doc_body


def index_single_chunk_with_retry(
    es,
    doc_id: int,
    chunk_index: int,
    doc_body: dict,
) -> None:
    es_doc_id = f"{doc_id}_{chunk_index + 1}"
    for attempt in range(1, ES_INDEX_MAX_RETRIES + 1):
        try:
            es.index(index=ELASTICSEARCH_INDEX, id=es_doc_id, document=doc_body)
            return
        except Exception as e:
            if attempt == ES_INDEX_MAX_RETRIES:
                logger.error(
                    "ES index permanently failed doc_id=%s chunk=%s after %s attempts: %s",
                    doc_id,
                    chunk_index,
                    ES_INDEX_MAX_RETRIES,
                    e,
                )
                raise
            delay = ES_INDEX_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "ES index failed doc_id=%s chunk=%s attempt=%s/%s, retrying in %.1fs: %s",
                doc_id,
                chunk_index,
                attempt,
                ES_INDEX_MAX_RETRIES,
                delay,
                e,
            )
            time.sleep(delay)


def index_all_chunks(
    es,
    embedding,
    doc_id: int,
    chunks: list[str],
    owner_id: Any,
    folder_id: Any,
    doc_name: str | None,
    *,
    with_retry: bool,
) -> None:
    for i, chunk in enumerate(chunks):
        vector = embedding.encode_text(chunk)
        body = build_chunk_document_body(
            doc_id, i, chunk, vector, owner_id, folder_id, doc_name
        )
        if with_retry:
            index_single_chunk_with_retry(es, doc_id, i, body)
        else:
            es.index(
                index=ELASTICSEARCH_INDEX,
                id=f"{doc_id}_{i + 1}",
                document=body,
            )
