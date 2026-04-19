import logging
from typing import Any

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE, ELASTICSEARCH_INDEX

logger = logging.getLogger(__name__)


def index_all_chunks(
    es,
    embedding,
    doc_id: int,
    chunks: list[str],
    owner_id: Any,
    doc_name: str | None,
) -> None:
    for i, chunk in enumerate(chunks):
        vector = embedding.encode_text(chunk)
        body = build_chunk_document_body(
            doc_id, i, chunk, vector, owner_id, doc_name
        )
        es.index(
            index=ELASTICSEARCH_INDEX,
            id=f"{doc_id}_{i + 1}",
            document=body,
        )


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


def delete_all_chunks_for_document(es, doc_id: int) -> None:
    try:
        es.delete_by_query(
            index=ELASTICSEARCH_INDEX,
            body={"query": {"term": {"document_id": str(doc_id)}}},
            refresh=True,
        )
        logger.info("Deleted prior Elasticsearch chunks for document_id=%s", doc_id)
    except Exception as e:
        logger.warning(
            "Failed to delete existing chunks for document_id=%s: %s", doc_id, e
        )


def build_chunk_document_body(
    doc_id: int,
    chunk_index: int,
    chunk: str,
    content_vector: list[float],
    owner_id: Any,
    doc_name: str | None,
) -> dict:
    doc_body: dict = {
        "chunk_id": f"{doc_id}_{chunk_index + 1}",
        "document_id": str(doc_id),
        "owner_id": str(owner_id) if owner_id is not None else None,
        "content": chunk,
        "content_vector": content_vector,
    }
    if doc_name is not None:
        doc_body["name"] = doc_name
    return doc_body
