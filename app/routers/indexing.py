import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import ELASTICSEARCH_INDEX
from app.core.es import es_client
from app.deps import get_embedding_service
from app.models.indexing import ReindexRequest, ReindexResponse
from app.services.embedding_service import EmbeddingService
from app.services.es_document_indexing import chunk_text, index_all_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reindex/{doc_id}", response_model=ReindexResponse)
def reindex_document(
    doc_id: int,
    req: ReindexRequest,
    embedding: EmbeddingService = Depends(get_embedding_service),
):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    es = es_client.get_client()

    try:
        es.delete_by_query(
            index=ELASTICSEARCH_INDEX,
            body={"query": {"term": {"document_id": str(doc_id)}}},
            refresh=True,
        )
    except Exception as e:
        logger.warning("Failed to delete existing chunks for doc_id=%s: %s", doc_id, e)

    chunks = chunk_text(req.text)
    index_all_chunks(
        es,
        embedding,
        doc_id,
        chunks,
        req.owner_id,
        req.folder_id,
        req.name,
        with_retry=False,
    )

    return ReindexResponse(success=True, chunks_indexed=len(chunks))
