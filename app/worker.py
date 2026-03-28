import json
import logging
from typing import List
import pika
import requests
from app.core.config import (
    BACKEND_BASE_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ELASTICSEARCH_INDEX,
    RABBITMQ_DOCUMENT_QUEUE,
    RABBITMQ_URL,
)
from app.core.es import es_client
from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.ocr_service import (
    ImageTooBlurryError,
    OcrService,
    UnsupportedFileTypeError,
    ocr_service,
)

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, embedding: EmbeddingService, ocr: OcrService) -> None:
        self._embedding = embedding
        self._ocr = ocr

    @staticmethod
    def compute_ocr_score(text: str) -> int:
        if not text:
            return 0
        total = len(text)
        clean = sum(1 for c in text if c.isalnum() or c.isspace())
        return int(clean / total * 100)

    @staticmethod
    def chunk_text(
        text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + size]
            chunks.append(" ".join(chunk_words))
            if i + size >= len(words):
                break
            i += size - overlap
        return chunks

    @staticmethod
    def update_processing_status(
        doc_id: int,
        status: str,
        score: int | None,
        error: str | None,
        extracted_text: str | None = None,
    ) -> None:
        base = (BACKEND_BASE_URL or "").rstrip("/")
        url = f"{base}/api/internal/documents/{doc_id}/processing-status"
        payload = {
            "processingStatus": status,
            "ocrQualityScore": score,
            "processingError": error,
        }
        if extracted_text is not None:
            payload["extractedText"] = extracted_text
        try:
            resp = requests.patch(url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(
                    "processing-status PATCH failed for doc_id=%s status=%s http=%s body=%s",
                    doc_id,
                    status,
                    resp.status_code,
                    (resp.text or "")[:500],
                )
        except Exception as e:
            logger.warning(
                "processing-status PATCH error for doc_id=%s status=%s: %s",
                doc_id,
                status,
                e,
            )

    @staticmethod
    def _ack_message(ch, method) -> None:
        ch.basic_ack(delivery_tag=method.delivery_tag)

    @staticmethod
    def _decode_message_body(body) -> dict:
        raw = bytes(body) if isinstance(body, memoryview) else body
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return json.loads(text)

    def _parse_message(self, body) -> tuple[int, str, dict]:
        msg = self._decode_message_body(body)
        doc_id = msg["doc_id"]
        file_url = msg["file_url"]
        return doc_id, file_url, msg

    def _handle_ocr_failure(
        self, ch, method, doc_id: int, message: str, log_msg: str
    ) -> None:
        logger.warning(log_msg, doc_id, message)
        self.update_processing_status(doc_id, "FAILED", 0, message)
        self._ack_message(ch, method)

    def _extract_text_or_fail(
        self, ch, method, doc_id: int, temp_path: str, file_type: str
    ):
        try:
            return self._ocr.run_ocr(temp_path, file_type)
        except ImageTooBlurryError as e:
            self._handle_ocr_failure(
                ch,
                method,
                doc_id,
                str(e),
                "OCR rejected blurry image doc_id=%s: %s",
            )
            return None
        except UnsupportedFileTypeError as e:
            self._handle_ocr_failure(
                ch,
                method,
                doc_id,
                str(e),
                "OCR rejected unsupported type doc_id=%s: %s",
            )
            return None

    def _build_doc_body(
        self,
        doc_id: int,
        chunk_index: int,
        chunk: str,
        owner_id,
        folder_id,
        doc_name,
    ) -> dict:
        vector = self._embedding.encode_text(chunk)
        doc_body = {
            "chunk_id": f"{doc_id}_{chunk_index}",
            "document_id": str(doc_id),
            "owner_id": str(owner_id) if owner_id is not None else None,
            "folder_id": str(folder_id) if folder_id is not None else None,
            "content": chunk,
            "content_vector": vector,
        }
        if doc_name is not None:
            doc_body["name"] = doc_name
        return doc_body

    def _index_chunks(
        self,
        doc_id: int,
        chunks: List[str],
        owner_id,
        folder_id,
        doc_name,
    ) -> None:
        es = es_client.get_client()
        for i, chunk in enumerate(chunks):
            doc_body = self._build_doc_body(
                doc_id=doc_id,
                chunk_index=i,
                chunk=chunk,
                owner_id=owner_id,
                folder_id=folder_id,
                doc_name=doc_name,
            )
            es.index(
                index=ELASTICSEARCH_INDEX,
                id=f"{doc_id}_{i + 1}",
                document=doc_body,
            )

    def process_message(self, ch, method, properties, body):
        try:
            doc_id, file_url, msg = self._parse_message(body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(
                "Skipping invalid queue message (expected UTF-8 JSON with doc_id, file_url): %s",
                e,
            )
            self._ack_message(ch, method)
            return

        file_type = msg.get("file_type", "application/octet-stream")
        owner_id = msg.get("owner_id")
        folder_id = msg.get("folder_id")
        doc_name = msg.get("name")

        temp_path: str | None = None

        try:
            logger.info(
                "Processing document doc_id=%s type=%s name=%r",
                doc_id,
                file_type,
                doc_name,
            )
            self.update_processing_status(doc_id, "PROCESSING", None, None)

            temp_path = self._ocr.download_to_temp(file_url, file_type)
            logger.debug("Downloaded to temp path=%s", temp_path)

            raw_text = self._extract_text_or_fail(
                ch=ch,
                method=method,
                doc_id=doc_id,
                temp_path=temp_path,
                file_type=file_type,
            )
            if raw_text is None:
                return

            if not raw_text.strip():
                logger.warning("OCR produced no text doc_id=%s", doc_id)
                self.update_processing_status(
                    doc_id,
                    "FAILED",
                    0,
                    "Could not extract content (OCR produced no text)",
                )
                self._ack_message(ch, method)
                return

            # Chunking + Embedding
            score = self.compute_ocr_score(raw_text)
            chunks = self.chunk_text(raw_text)
            logger.info(
                "OCR done doc_id=%s ocr_score=%s chunks=%s chars=%s",
                doc_id,
                score,
                len(chunks),
                len(raw_text),
            )

            self._index_chunks(doc_id, chunks, owner_id, folder_id, doc_name)

            # ACK
            logger.info(
                "Indexed doc_id=%s into %s (%s chunks)",
                doc_id,
                ELASTICSEARCH_INDEX,
                len(chunks),
            )
            self.update_processing_status(
                doc_id, "COMPLETED", score, None, extracted_text=raw_text
            )
            self._ack_message(ch, method)
        except Exception as e:
            logger.exception("Processing failed doc_id=%s", doc_id)
            self.update_processing_status(doc_id, "FAILED", 0, str(e))
            self._ack_message(ch, method)
        finally:
            if temp_path:
                self._ocr.cleanup_temp(temp_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info(
        "Worker starting queue=%s elasticsearch_index=%s",
        RABBITMQ_DOCUMENT_QUEUE,
        ELASTICSEARCH_INDEX,
    )

    indexer = DocumentIndexer(embedding_service, ocr_service)

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_DOCUMENT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=RABBITMQ_DOCUMENT_QUEUE, on_message_callback=indexer.process_message
    )
    logger.info("Consuming messages on %s (prefetch=1)", RABBITMQ_DOCUMENT_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()
