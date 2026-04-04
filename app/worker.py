import json
import logging
import time
import pika
import requests
from prometheus_client import start_http_server as start_metrics_server

from app.core.config import (
    BACKEND_BASE_URL,
    ELASTICSEARCH_INDEX,
    RABBITMQ_DOCUMENT_QUEUE,
    RABBITMQ_URL,
)
from app.core.es import es_client
from app.core.metrics import (
    DOCS_PROCESSED,
    PROCESSING_DURATION,
    QUALITY_SCORE,
    VALIDATION_FAILURES,
)
from app.services.data_validation_service import (
    DataValidationService,
    data_validation_service,
)
from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.es_document_indexing import (
    chunk_text,
    delete_all_chunks_for_document,
    index_all_chunks,
)
from app.services.ocr_monitoring_service import (
    OcrMonitoringService,
    ocr_monitoring_service,
)
from app.services.ocr_service import (
    ImageTooBlurryError,
    OcrService,
    UnsupportedFileTypeError,
    ocr_service,
)

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        embedding: EmbeddingService,
        ocr: OcrService,
        validation: DataValidationService | None = None,
        monitoring: OcrMonitoringService | None = None,
    ) -> None:
        self._embedding = embedding
        self._ocr = ocr
        self._validation = validation or data_validation_service
        self._monitoring = monitoring or ocr_monitoring_service

    @staticmethod
    def compute_ocr_score(text: str) -> int:
        if not text:
            return 0
        total = len(text)
        clean = sum(1 for c in text if c.isalnum() or c.isspace())
        return int(clean / total * 100)

    @staticmethod
    def update_processing_status(
        doc_id: int,
        status: str,
        score: int | None,
        error: str | None,
        extracted_text: str | None = None,
        validation_report: str | None = None,
        ocr_metrics: str | None = None,
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
        if validation_report is not None:
            payload["validationReport"] = validation_report
        if ocr_metrics is not None:
            payload["ocrMetrics"] = ocr_metrics
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

    def process_message(self, ch, method, properties, body):
        """RabbitMQ callback: download → validate → OCR → metrics → ES index → backend status."""
        # Parse queue payload (doc_id, file_url, metadata).
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
        t0 = time.monotonic()

        try:
            # Mark document as processing in the backend API.
            logger.info(
                "Processing document doc_id=%s type=%s name=%r",
                doc_id,
                file_type,
                doc_name,
            )
            self.update_processing_status(doc_id, "PROCESSING", None, None)

            # Fetch file from URL into a temp path for OCR.
            temp_path = self._ocr.download_to_temp(file_url, file_type)
            logger.debug("Downloaded to temp path=%s", temp_path)

            # Pre-OCR validation (image/PDF checks); fail fast with validation report.
            validation_report = self._validation.validate(temp_path, file_type, doc_id)
            report_json = validation_report.model_dump_json()
            if not validation_report.overall_passed:
                VALIDATION_FAILURES.inc()
                DOCS_PROCESSED.labels(status="failed").inc()
                failed_msgs = "; ".join(
                    c.message or c.name
                    for c in validation_report.checks
                    if not c.passed
                )
                self.update_processing_status(
                    doc_id,
                    "FAILED",
                    0,
                    f"Data validation failed: {failed_msgs}",
                    validation_report=report_json,
                )
                self._ack_message(ch, method)
                return

            # OCR: extract text; specific OCR errors update FAILED and ack inside helper.
            raw_text = self._extract_text_or_fail(
                ch=ch,
                method=method,
                doc_id=doc_id,
                temp_path=temp_path,
                file_type=file_type,
            )
            if raw_text is None:
                DOCS_PROCESSED.labels(status="failed").inc()
                return

            # Empty text after OCR is treated as a hard failure.
            if not raw_text.strip():
                logger.warning("OCR produced no text doc_id=%s", doc_id)
                DOCS_PROCESSED.labels(status="failed").inc()
                self.update_processing_status(
                    doc_id,
                    "FAILED",
                    0,
                    "Could not extract content (OCR produced no text)",
                    validation_report=report_json,
                )
                self._ack_message(ch, method)
                return

            # Quality metrics, histograms, and optional alerts.
            metrics = self._monitoring.compute_detailed_metrics(raw_text)
            metrics_json = json.dumps(metrics)
            self._monitoring.check_quality_alert(metrics, doc_id)
            score = metrics.get("ocr_quality_score", self.compute_ocr_score(raw_text))
            QUALITY_SCORE.observe(score)

            # Chunk text for embedding + Elasticsearch indexing.
            chunks = chunk_text(raw_text)
            logger.info(
                "OCR done doc_id=%s ocr_score=%s chunks=%s chars=%s",
                doc_id,
                score,
                len(chunks),
                len(raw_text),
            )

            # Elasticsearch: delete existing chunks for this document_id, then index new chunks.
            # On ES error: FAILED but keep extracted text and metrics for debugging/ops.
            try:
                es = es_client.get_client()
                delete_all_chunks_for_document(es, doc_id)
                index_all_chunks(
                    es,
                    self._embedding,
                    doc_id,
                    chunks,
                    owner_id,
                    folder_id,
                    doc_name,
                )
            except Exception as idx_err:
                logger.error(
                    "Elasticsearch indexing failed doc_id=%s: %s", doc_id, idx_err
                )
                DOCS_PROCESSED.labels(status="index_failed").inc()
                self.update_processing_status(
                    doc_id,
                    "FAILED",
                    score,
                    f"Elasticsearch indexing failed: {idx_err}",
                    extracted_text=raw_text,
                    validation_report=report_json,
                    ocr_metrics=metrics_json,
                )
                self._ack_message(ch, method)
                return

            # Success: indexed; persist COMPLETED and full payload to the backend.
            logger.info(
                "Indexed doc_id=%s into %s (%s chunks)",
                doc_id,
                ELASTICSEARCH_INDEX,
                len(chunks),
            )
            DOCS_PROCESSED.labels(status="completed").inc()
            self.update_processing_status(
                doc_id,
                "COMPLETED",
                score,
                None,
                extracted_text=raw_text,
                validation_report=report_json,
                ocr_metrics=metrics_json,
            )
            self._ack_message(ch, method)
        except Exception as e:
            # Unexpected errors anywhere above (download, validation bug, etc.).
            logger.exception("Processing failed doc_id=%s", doc_id)
            DOCS_PROCESSED.labels(status="failed").inc()
            self.update_processing_status(doc_id, "FAILED", 0, str(e))
            self._ack_message(ch, method)
        finally:
            # Observe wall-clock duration for this message; always delete temp file.
            PROCESSING_DURATION.observe(time.monotonic() - t0)
            if temp_path:
                self._ocr.cleanup_temp(temp_path)


def main():
    start_metrics_server(8001)
    logger.info("Prometheus metrics server started on :8001")

    logger.info(
        "Worker starting queue=%s elasticsearch_index=%s",
        RABBITMQ_DOCUMENT_QUEUE,
        ELASTICSEARCH_INDEX,
    )

    indexer = DocumentIndexer(
        embedding_service, ocr_service, data_validation_service, ocr_monitoring_service
    )

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
