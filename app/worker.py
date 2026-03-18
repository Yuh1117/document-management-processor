import json
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
from app.services.ocr_service import ImageTooBlurryError, OcrService, ocr_service


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
        doc_id: int, status: str, score: int | None, error: str | None
    ) -> None:
        url = f"{BACKEND_BASE_URL}/internal/documents/{doc_id}/processing-status"
        payload = {
            "processingStatus": status,
            "ocrQualityScore": score,
            "processingError": error,
        }
        try:
            requests.patch(url, json=payload, timeout=10)
        except Exception:
            pass

    def process_message(self, ch, method, properties, body):
        msg = json.loads(body)
        doc_id = msg["doc_id"]
        file_url = msg["file_url"]
        file_type = msg.get("file_type", "application/octet-stream")
        owner_id = msg.get("owner_id")
        folder_id = msg.get("folder_id")
        doc_name = msg.get("name")

        temp_path: str | None = None

        try:
            self.update_processing_status(doc_id, "PROCESSING", None, None)

            temp_path = self._ocr.download_to_temp(file_url, file_type)

            # MLOps - Data Validation (blur) + OCR
            try:
                raw_text = self._ocr.run_ocr(temp_path, file_type)
            except ImageTooBlurryError as e:
                self.update_processing_status(doc_id, "FAILED", 0, str(e))
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            if not raw_text.strip():
                self.update_processing_status(
                    doc_id,
                    "FAILED",
                    0,
                    "Could not extract content (OCR produced no text)",
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Chunking + Embedding
            score = self.compute_ocr_score(raw_text)
            chunks = self.chunk_text(raw_text)

            # Index Elasticsearch
            es = es_client.get_client()
            for i, chunk in enumerate(chunks):
                vector = self._embedding.encode_text(chunk)
                doc_body = {
                    "chunk_id": f"{doc_id}_{i}",
                    "document_id": str(doc_id),
                    "owner_id": str(owner_id) if owner_id is not None else None,
                    "folder_id": str(folder_id) if folder_id is not None else None,
                    "content": chunk,
                    "content_vector": vector,
                }
                if doc_name is not None:
                    doc_body["name"] = doc_name
                es.index(
                    index=ELASTICSEARCH_INDEX,
                    id=f"{doc_id}_{i}",
                    document=doc_body,
                )

            # ACK
            self.update_processing_status(doc_id, "COMPLETED", score, None)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            self.update_processing_status(doc_id, "FAILED", 0, str(e))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        finally:
            if temp_path:
                self._ocr.cleanup_temp(temp_path)


def main():
    indexer = DocumentIndexer(embedding_service, ocr_service)

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_DOCUMENT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=RABBITMQ_DOCUMENT_QUEUE, on_message_callback=indexer.process_message
    )
    channel.start_consuming()


if __name__ == "__main__":
    main()
