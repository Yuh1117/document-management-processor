import json
import logging
import time
from dataclasses import dataclass
from typing import Any
import pika
from prometheus_client import start_http_server as start_metrics_server
from app.core.config import (
    ELASTICSEARCH_INDEX,
    RABBITMQ_DOCUMENT_QUEUE,
    RABBITMQ_PROCESSING_RESULT_QUEUE,
    RABBITMQ_URL,
)
from app.core.es import es_client
from app.core.metrics import (
    DOCS_PROCESSED,
    PROCESSING_DURATION,
    QUALITY_SCORE,
    VALIDATION_FAILURES,
)
from app.services.embedding_service import EmbeddingService, embedding_service
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
from app.utils.elasticsearch.indexing import (
    chunk_text,
    delete_all_chunks_for_document,
    index_all_chunks,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMessage:
    doc_id: int
    file_url: str
    file_type: str = "application/octet-stream"
    owner_id: str | None = None
    name: str | None = None

    @classmethod
    def from_body(cls, body: bytes | bytearray | memoryview) -> "DocumentMessage":
        raw = bytes(body) if isinstance(body, memoryview) else body
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        msg: dict[str, Any] = json.loads(text)
        return cls(
            doc_id=msg["doc_id"],
            file_url=msg["file_url"],
            file_type=msg.get("file_type", "application/octet-stream"),
            owner_id=msg.get("owner_id"),
            name=msg.get("name"),
        )


class StatusPublisher:
    """Publishes processing-status updates to the result queue."""

    def __init__(self, channel, queue: str = RABBITMQ_PROCESSING_RESULT_QUEUE) -> None:
        self._ch = channel
        self._queue = queue

    def publish(
        self,
        doc_id: int,
        status: str,
        *,
        processing_report: str | None = None,
        extracted_text: str | None = None,
        ocr_metrics: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"documentId": doc_id, "processingStatus": status}
        if processing_report is not None:
            payload["processingReport"] = processing_report
        if extracted_text is not None:
            payload["extractedText"] = extracted_text
        if ocr_metrics is not None:
            payload["ocrMetrics"] = ocr_metrics
        try:
            self._ch.basic_publish(
                exchange="",
                routing_key=self._queue,
                body=json.dumps(payload),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                ),
            )
        except Exception as exc:
            logger.warning(
                "Status publish failed doc_id=%s status=%s: %s", doc_id, status, exc
            )

    def completed(self, doc_id: int, extracted_text: str, ocr_metrics: str) -> None:
        self.publish(
            doc_id, "COMPLETED", extracted_text=extracted_text, ocr_metrics=ocr_metrics
        )

    def failed(
        self,
        doc_id: int,
        message: str,
        report_extra: dict | None = None,
        extracted_text: str | None = None,
        ocr_metrics: str | None = None,
    ) -> None:
        report = {"message": message}
        if report_extra:
            report.update(report_extra)
        self.publish(
            doc_id,
            "FAILED",
            processing_report=json.dumps(report),
            extracted_text=extracted_text,
            ocr_metrics=ocr_metrics,
        )

    def processing(self, doc_id: int) -> None:
        self.publish(doc_id, "PROCESSING")


class DocumentIndexer:
    def __init__(
        self,
        embedding: EmbeddingService,
        ocr: OcrService,
        monitoring: OcrMonitoringService | None = None,
    ) -> None:
        self.embedding = embedding
        self.ocr = ocr
        self.monitoring = monitoring or ocr_monitoring_service

    def process_message(self, ch, method, properties, body) -> None:
        try:
            doc = DocumentMessage.from_body(body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Skipping invalid queue message: %s", exc)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        publisher = StatusPublisher(ch)
        t0 = time.monotonic()
        temp_path: str | None = None

        try:
            self.process(doc, publisher, temp_path_ref := [None])
            temp_path = temp_path_ref[0]
        except Exception:
            logger.exception("Unexpected error doc_id=%s", doc.doc_id)
            DOCS_PROCESSED.labels(status="failed").inc()
            publisher.failed(doc.doc_id, "Unexpected internal error")
        finally:
            PROCESSING_DURATION.observe(time.monotonic() - t0)
            if temp_path:
                self.ocr.cleanup_temp(temp_path)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def process(
        self, doc: DocumentMessage, publisher: StatusPublisher, temp_path_ref: list
    ) -> None:
        logger.info(
            "Processing doc_id=%s type=%s name=%r", doc.doc_id, doc.file_type, doc.name
        )
        publisher.processing(doc.doc_id)

        temp_path = self.ocr.download_to_temp(doc.file_url, doc.file_type)
        temp_path_ref[0] = temp_path
        logger.debug("Downloaded to temp_path=%s", temp_path)

        if not self.validate(doc, publisher, temp_path):
            return

        raw_text = self.extract_text(doc, publisher, temp_path)
        if raw_text is None:
            return

        metrics = self._compute_metrics(doc, raw_text)
        self._index(doc, publisher, raw_text, metrics)

    def validate(
        self, doc: DocumentMessage, publisher: StatusPublisher, temp_path: str
    ) -> bool:
        result = self.ocr.validate(temp_path, doc.file_type, doc.doc_id)
        if result.overall_passed:
            return True

        VALIDATION_FAILURES.inc()
        DOCS_PROCESSED.labels(status="failed").inc()
        failed_checks = [
            {"name": c.name, "passed": False, "message": c.message}
            for c in result.checks
            if not c.passed
        ]
        summary = "; ".join(c["message"] or c["name"] for c in failed_checks)
        publisher.failed(
            doc.doc_id,
            f"Data validation failed: {summary}",
            report_extra={"checks": failed_checks},
        )
        return False

    def extract_text(
        self, doc: DocumentMessage, publisher: StatusPublisher, temp_path: str
    ) -> str | None:
        try:
            raw_text = self.ocr.run_ocr(temp_path, doc.file_type)
        except ImageTooBlurryError as exc:
            logger.warning("Blurry image doc_id=%s: %s", doc.doc_id, exc)
            publisher.failed(doc.doc_id, str(exc))
            DOCS_PROCESSED.labels(status="failed").inc()
            return None
        except UnsupportedFileTypeError as exc:
            logger.warning("Unsupported type doc_id=%s: %s", doc.doc_id, exc)
            publisher.failed(doc.doc_id, str(exc))
            DOCS_PROCESSED.labels(status="failed").inc()
            return None

        if not raw_text.strip():
            logger.warning("OCR produced no text doc_id=%s", doc.doc_id)
            publisher.failed(
                doc.doc_id, "Could not extract content (OCR produced no text)"
            )
            DOCS_PROCESSED.labels(status="failed").inc()
            return None

        return raw_text

    def _compute_metrics(self, doc: DocumentMessage, raw_text: str) -> dict:
        metrics = self.monitoring.compute_detailed_metrics(raw_text)
        self.monitoring.check_quality_alert(metrics, doc.doc_id)
        QUALITY_SCORE.observe(metrics["ocr_quality_score"])
        return metrics

    def _index(
        self,
        doc: DocumentMessage,
        publisher: StatusPublisher,
        raw_text: str,
        metrics: dict,
    ) -> None:
        chunks = chunk_text(raw_text)
        metrics_json = json.dumps(metrics)
        logger.info(
            "OCR done doc_id=%s score=%s chunks=%s chars=%s",
            doc.doc_id,
            metrics["ocr_quality_score"],
            len(chunks),
            len(raw_text),
        )

        try:
            es = es_client.get_client()
            delete_all_chunks_for_document(es, doc.doc_id)
            index_all_chunks(
                es, self.embedding, doc.doc_id, chunks, doc.owner_id, doc.name
            )
        except Exception as exc:
            logger.error("Elasticsearch indexing failed doc_id=%s: %s", doc.doc_id, exc)
            DOCS_PROCESSED.labels(status="index_failed").inc()
            publisher.publish(
                doc.doc_id,
                "FAILED",
                processing_report=json.dumps(
                    {"message": f"Elasticsearch indexing failed: {exc}"}
                ),
                extracted_text=raw_text,
                ocr_metrics=metrics_json,
            )
            publisher.failed(
                doc.doc_id,
                f"Elasticsearch indexing failed: {exc}",
                extracted_text=raw_text,
                ocr_metrics=metrics_json,
            )
            return

        logger.info(
            "Indexed doc_id=%s into %s (%s chunks)",
            doc.doc_id,
            ELASTICSEARCH_INDEX,
            len(chunks),
        )
        DOCS_PROCESSED.labels(status="completed").inc()
        publisher.completed(doc.doc_id, raw_text, metrics_json)


def main() -> None:
    start_metrics_server(8001)
    logger.info("Prometheus metrics server started on :8001")
    logger.info(
        "Worker starting queue=%s index=%s",
        RABBITMQ_DOCUMENT_QUEUE,
        ELASTICSEARCH_INDEX,
    )

    indexer = DocumentIndexer(embedding_service, ocr_service, ocr_monitoring_service)

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_DOCUMENT_QUEUE, durable=True)
    channel.queue_declare(queue=RABBITMQ_PROCESSING_RESULT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=RABBITMQ_DOCUMENT_QUEUE, on_message_callback=indexer.process_message
    )

    logger.info("Consuming messages on %s (prefetch=1)", RABBITMQ_DOCUMENT_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()
