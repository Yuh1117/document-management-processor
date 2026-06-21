import logging
from fastapi import APIRouter, Depends
from app.core.es import es_client
from app.deps import verify_api_key
from app.utils.elasticsearch.indexing import delete_all_chunks_for_document

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.delete("/index/{doc_id}")
def delete_document_index(doc_id: int):
    es = es_client.get_client()
    delete_all_chunks_for_document(es, doc_id)
    return {"success": True}
