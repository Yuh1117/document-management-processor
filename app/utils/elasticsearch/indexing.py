import logging
import re

from elasticsearch.helpers import bulk, scan

from app.core.config import ELASTICSEARCH_INDEX
from app.constants.defaults import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def index_all_chunks(
    es,
    embedding,
    doc_id: str,
    chunks: list[str],
    owner_id: str | None,
    doc_name: str | None,
    mime_type: str | None = None,
    file_size: int | None = None,
) -> None:
    for i, chunk in enumerate(chunks):
        vector = embedding.encode_text(chunk)
        body = build_chunk_document_body(
            doc_id, i, chunk, vector, owner_id, doc_name, mime_type, file_size
        )
        es.index(
            index=ELASTICSEARCH_INDEX,
            id=f"{doc_id}_{i + 1}",
            document=body,
        )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []

        current.extend(words)

        while len(current) >= size:
            chunks.append(" ".join(current[:size]))
            current = current[size - overlap :] if overlap > 0 else current[size:]

    if current:
        chunks.append(" ".join(current))

    return chunks


def delete_all_chunks_for_document(es, doc_id: str) -> None:
    try:
        es.delete_by_query(
            index=ELASTICSEARCH_INDEX,
            body={"query": {"term": {"document_id": str(doc_id)}}},
        )
        logger.info("Deleted prior Elasticsearch chunks for document_id=%s", doc_id)
    except Exception as e:
        logger.warning(
            "Failed to delete existing chunks for document_id=%s: %s", doc_id, e
        )


def update_document_name(es, doc_id: str, name: str) -> int:
    resp = es.update_by_query(
        index=ELASTICSEARCH_INDEX,
        query={"term": {"document_id": str(doc_id)}},
        script={
            "source": "ctx._source.name = params.name",
            "params": {"name": name},
        },
        conflicts="proceed",
        refresh=True,
    )
    return int(resp.get("updated", 0))


def copy_document_chunks(
    es, source_id: str, new_id: str, name: str, owner_id: str | None
) -> int:
    actions = []
    for hit in scan(
        es,
        index=ELASTICSEARCH_INDEX,
        query={"query": {"term": {"document_id": str(source_id)}}},
    ):
        chunk_no = hit["_id"].rsplit("_", 1)[-1]
        chunk_id = f"{new_id}_{chunk_no}"
        body = dict(hit["_source"])
        body.update(
            chunk_id=chunk_id,
            document_id=str(new_id),
            name=name,
            owner_id=str(owner_id) if owner_id is not None else None,
        )
        actions.append({"_index": ELASTICSEARCH_INDEX, "_id": chunk_id, "_source": body})

    if not actions:
        return 0

    copied, _ = bulk(es, actions, refresh=True)
    return copied


def build_chunk_document_body(
    doc_id: str,
    chunk_index: int,
    chunk: str,
    content_vector: list[float],
    owner_id: str | None,
    doc_name: str | None,
    mime_type: str | None = None,
    file_size: int | None = None,
) -> dict:
    doc_body: dict = {
        "chunk_id": f"{doc_id}_{chunk_index + 1}",
        "document_id": str(doc_id),
        "owner_id": str(owner_id) if owner_id is not None else None,
        "mime_type": mime_type,
        "file_size": file_size,
        "content": chunk,
        "content_vector": content_vector,
    }
    if doc_name is not None:
        doc_body["name"] = doc_name
    return doc_body
