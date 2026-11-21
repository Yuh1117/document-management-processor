import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from app.core.config import (
    ELASTICSEARCH_INDEX,
)
from app.core.es import es_client
from app.core.metrics import (
    DOCS_PROCESSED,
    PROCESSING_DURATION,
    QUALITY_SCORE,
    VALIDATION_FAILURES,
)
from app.embedding.service import EmbeddingService
from app.extraction.errors import ImageTooBlurryError
from app.extraction.service import ExtractionService
from app.extraction.text_quality import TextQualityService, text_quality_service
from app.indexing.chunking import chunk_text
from app.indexing.es_indexing import delete_all_chunks_for_document, index_all_chunks
from app.worker.constants import WORKER_THREADS
from app.worker.message import DocumentMessage
from app.worker.publisher import StatusPublisher

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
        self,
        embedding: EmbeddingService,
        extraction: ExtractionService,
        monitoring: TextQualityService | None = None,
    ) -> None:
        self.embedding = embedding
        self.extraction = extraction
        self.monitoring = monitoring or text_quality_service
        self._executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)

    def process_message(self, connection, ch, method, _properties, body) -> None:
        try:
            doc = DocumentMessage.from_body(body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Skipping invalid queue message: %s", exc)
            connection.add_callback_threadsafe(
                lambda: ch.basic_ack(delivery_tag=method.delivery_tag)
            )
            return

        publisher = StatusPublisher(connection, ch)
        self._executor.submit(
            self._process_in_thread, connection, ch, method, doc, publisher
        )

    def _process_in_thread(self, connection, ch, method, doc, publisher) -> None:
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
                self.extraction.cleanup_temp(temp_path)
            connection.add_callback_threadsafe(
                lambda: ch.basic_ack(delivery_tag=method.delivery_tag)
            )

    def process(
        self, doc: DocumentMessage, publisher: StatusPublisher, temp_path_ref: list
    ) -> None:
        logger.info(
            "Processing doc_id=%s type=%s name=%r", doc.doc_id, doc.file_type, doc.name
        )

        if not self.extraction.is_supported(doc.file_type):
            logger.warning(
                "Unsupported file type doc_id=%s type=%s", doc.doc_id, doc.file_type
            )
            DOCS_PROCESSED.labels(status="failed").inc()
            publisher.failed(doc.doc_id, f"Unsupported file type: {doc.file_type}")
            return

        publisher.processing(doc.doc_id)

        temp_path = self.extraction.download_to_temp(doc.file_url, doc.file_type)
        temp_path_ref[0] = temp_path
        logger.debug("Downloaded to temp_path=%s", temp_path)

        if not self.validate(doc, publisher, temp_path):
            return

        raw_text = self.extract_text(doc, publisher, temp_path)
        if raw_text is None:
            return

        metrics = self.compute_metrics(doc, raw_text)
        self.index(doc, publisher, raw_text, metrics)

    def validate(
        self, doc: DocumentMessage, publisher: StatusPublisher, temp_path: str
    ) -> bool:
        result = self.extraction.validate(temp_path, doc.file_type, doc.doc_id)
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
            raw_text = self.extraction.extract_text(temp_path, doc.file_type)
        except ImageTooBlurryError as exc:
            logger.warning("Blurry image doc_id=%s: %s", doc.doc_id, exc)
            publisher.failed(doc.doc_id, str(exc))
            DOCS_PROCESSED.labels(status="failed").inc()
            return None

        if not raw_text.strip():
            logger.warning("Produced no text doc_id=%s", doc.doc_id)
            publisher.failed(doc.doc_id, "Could not extract content (Produced no text)")
            DOCS_PROCESSED.labels(status="failed").inc()
            return None

        return raw_text

    def compute_metrics(self, doc: DocumentMessage, raw_text: str) -> dict:
        metrics = self.monitoring.compute_detailed_metrics(raw_text)
        self.monitoring.check_quality_alert(metrics, doc.doc_id)
        QUALITY_SCORE.observe(metrics["text_quality_score"])
        return metrics

    def index(
        self,
        doc: DocumentMessage,
        publisher: StatusPublisher,
        raw_text: str,
        metrics: dict,
    ) -> None:
        chunks = chunk_text(raw_text)
        metrics_json = json.dumps(metrics)
        logger.info(
            "Extract done doc_id=%s score=%s chunks=%s chars=%s",
            doc.doc_id,
            metrics["text_quality_score"],
            len(chunks),
            len(raw_text),
        )

        try:
            es = es_client.get_client()
            delete_all_chunks_for_document(es, doc.doc_id)
            index_all_chunks(
                es,
                self.embedding,
                doc.doc_id,
                chunks,
                doc.owner_id,
                doc.name,
                doc.file_type,
                doc.file_size,
            )
        except Exception as exc:
            logger.error("Elasticsearch indexing failed doc_id=%s: %s", doc.doc_id, exc)
            DOCS_PROCESSED.labels(status="failed").inc()
            publisher.publish(
                doc.doc_id,
                "FAILED",
                processing_report=json.dumps(
                    {"message": f"Elasticsearch indexing failed: {exc}"}
                ),
                extracted_text=raw_text,
                processing_metrics=metrics_json,
            )
            publisher.failed(
                doc.doc_id,
                f"Elasticsearch indexing failed: {exc}",
                extracted_text=raw_text,
                processing_metrics=metrics_json,
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
