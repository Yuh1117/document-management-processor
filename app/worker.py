import json
import logging
import time
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
from app.utils.elasticsearch.indexing import (
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
        monitoring: OcrMonitoringService | None = None,
    ) -> None:
        self.embedding = embedding
        self.ocr = ocr
        self.monitoring = monitoring or ocr_monitoring_service

    def process_message(self, ch, method, properties, body):
        """RabbitMQ callback: download → validate → OCR → metrics → ES index → backend status."""
        try:
            doc_id, file_url, msg = self.parse_message(body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(
                "Skipping invalid queue message (expected UTF-8 JSON with doc_id, file_url): %s",
                e,
            )
            self.ack_message(ch, method)
            return

        file_type = msg.get("file_type", "application/octet-stream")
        owner_id = msg.get("owner_id")
        folder_id = msg.get("folder_id")
        doc_name = msg.get("name")

        temp_path: str | None = None
        t0 = time.monotonic()

        try:
            logger.info(
                "Processing document doc_id=%s type=%s name=%r",
                doc_id,
                file_type,
                doc_name,
            )
            self.update_processing_status(ch, doc_id, "PROCESSING")

            temp_path = self.ocr.download_to_temp(file_url, file_type)
            logger.debug("Downloaded to temp path=%s", temp_path)

            validation_result = self.ocr.validate(temp_path, file_type, doc_id)
            if not validation_result.overall_passed:
                VALIDATION_FAILURES.inc()
                DOCS_PROCESSED.labels(status="failed").inc()
                failed_checks = [
                    {"name": c.name, "passed": False, "message": c.message}
                    for c in validation_result.checks
                    if not c.passed
                ]
                failed_summary = "; ".join(c["message"] or c["name"] for c in failed_checks)
                processing_report = json.dumps({
                    "message": f"Data validation failed: {failed_summary}",
                    "checks": failed_checks,
                })
                self.update_processing_status(
                    ch,
                    doc_id,
                    "FAILED",
                    processing_report=processing_report,
                )
                self.ack_message(ch, method)
                return

            raw_text = self.extract_text_or_fail(
                ch=ch,
                method=method,
                doc_id=doc_id,
                temp_path=temp_path,
                file_type=file_type,
            )
            if raw_text is None:
                DOCS_PROCESSED.labels(status="failed").inc()
                return

            if not raw_text.strip():
                logger.warning("OCR produced no text doc_id=%s", doc_id)
                DOCS_PROCESSED.labels(status="failed").inc()
                self.update_processing_status(
                    ch,
                    doc_id,
                    "FAILED",
                    processing_report=json.dumps({"message": "Could not extract content (OCR produced no text)"}),
                )
                self.ack_message(ch, method)
                return

            metrics = self.monitoring.compute_detailed_metrics(raw_text)
            metrics_json = json.dumps(metrics)
            self.monitoring.check_quality_alert(metrics, doc_id)
            score = metrics["ocr_quality_score"]
            QUALITY_SCORE.observe(score)

            chunks = chunk_text(raw_text)
            logger.info(
                "OCR done doc_id=%s ocr_score=%s chunks=%s chars=%s",
                doc_id,
                score,
                len(chunks),
                len(raw_text),
            )

            try:
                es = es_client.get_client()
                delete_all_chunks_for_document(es, doc_id)
                index_all_chunks(
                    es,
                    self.embedding,
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
                    ch,
                    doc_id,
                    "FAILED",
                    processing_report=json.dumps({"message": f"Elasticsearch indexing failed: {idx_err}"}),
                    extracted_text=raw_text,
                    ocr_metrics=metrics_json,
                )
                self.ack_message(ch, method)
                return

            logger.info(
                "Indexed doc_id=%s into %s (%s chunks)",
                doc_id,
                ELASTICSEARCH_INDEX,
                len(chunks),
            )
            DOCS_PROCESSED.labels(status="completed").inc()
            self.update_processing_status(
                ch,
                doc_id,
                "COMPLETED",
                extracted_text=raw_text,
                ocr_metrics=metrics_json,
            )
            self.ack_message(ch, method)
        except Exception as e:
            logger.exception("Processing failed doc_id=%s", doc_id)
            DOCS_PROCESSED.labels(status="failed").inc()
            self.update_processing_status(
                ch,
                doc_id,
                "FAILED",
                processing_report=json.dumps({"message": str(e)}),
            )
            self.ack_message(ch, method)
        finally:
            PROCESSING_DURATION.observe(time.monotonic() - t0)
            if temp_path:
                self.ocr.cleanup_temp(temp_path)

    @staticmethod
    def update_processing_status(
        ch,
        doc_id: int,
        status: str,
        processing_report: str | None = None,
        extracted_text: str | None = None,
        ocr_metrics: str | None = None,
    ) -> None:
        payload = {"documentId": doc_id, "processingStatus": status}
        if processing_report is not None:
            payload["processingReport"] = processing_report
        if extracted_text is not None:
            payload["extractedText"] = extracted_text
        if ocr_metrics is not None:
            payload["ocrMetrics"] = ocr_metrics
        try:
            ch.basic_publish(
                exchange="",
                routing_key=RABBITMQ_PROCESSING_RESULT_QUEUE,
                body=json.dumps(payload),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                ),
            )
        except Exception as e:
            logger.warning(
                "processing-status publish failed for doc_id=%s status=%s: %s",
                doc_id,
                status,
                e,
            )

    def extract_text_or_fail(
        self, ch, method, doc_id: int, temp_path: str, file_type: str
    ):
        try:
            return self.ocr.run_ocr(temp_path, file_type)
        except ImageTooBlurryError as e:
            self.handle_ocr_failure(
                ch,
                method,
                doc_id,
                str(e),
                "OCR rejected blurry image doc_id=%s: %s",
            )
            return None
        except UnsupportedFileTypeError as e:
            self.handle_ocr_failure(
                ch,
                method,
                doc_id,
                str(e),
                "OCR rejected unsupported type doc_id=%s: %s",
            )
            return None

    def handle_ocr_failure(
        self, ch, method, doc_id: int, message: str, log_msg: str
    ) -> None:
        logger.warning(log_msg, doc_id, message)
        self.update_processing_status(
            ch, doc_id, "FAILED", processing_report=json.dumps({"message": message})
        )
        self.ack_message(ch, method)

    def parse_message(self, body) -> tuple[int, str, dict]:
        msg = self.decode_message_body(body)
        doc_id = msg["doc_id"]
        file_url = msg["file_url"]
        return doc_id, file_url, msg

    @staticmethod
    def decode_message_body(body) -> dict:
        raw = bytes(body) if isinstance(body, memoryview) else body
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return json.loads(text)

    @staticmethod
    def ack_message(ch, method) -> None:
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    start_metrics_server(8001)
    logger.info("Prometheus metrics server started on :8001")

    logger.info(
        "Worker starting queue=%s elasticsearch_index=%s",
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
