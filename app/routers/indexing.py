import logging

from fastapi import APIRouter, Depends

from app.core.es import es_client
from app.deps import verify_api_key
from app.models.indexing import CopyRequest, RenameRequest
from app.utils.elasticsearch.indexing import (
    copy_document_chunks,
    delete_all_chunks_for_document,
    update_document_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.delete("/index/{doc_id}")
def delete_document_index(doc_id: str):
    es = es_client.get_client()
    delete_all_chunks_for_document(es, doc_id)
    return {"success": True}


@router.post("/index/{doc_id}/name")
def rename_document_index(doc_id: str, req: RenameRequest):
    es = es_client.get_client()
    chunks_updated = update_document_name(es, doc_id, req.name)
    return {"success": True, "chunks_updated": chunks_updated}


@router.post("/index/{doc_id}/copy")
def copy_document_index(doc_id: str, req: CopyRequest):
    es = es_client.get_client()
    chunks_copied = copy_document_chunks(
        es, doc_id, req.new_id, req.name, req.owner_id
    )
    return {"success": True, "chunks_copied": chunks_copied}
