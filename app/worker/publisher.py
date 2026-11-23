import json
import logging
from typing import Any

import pika

from app.core.config import (
    RABBITMQ_PROCESSING_RESULT_QUEUE,
)

logger = logging.getLogger(__name__)


class StatusPublisher:
    """Publishes processing-status updates to the result queue."""

    def __init__(
        self,
        connection: pika.BlockingConnection,
        channel,
        queue: str = RABBITMQ_PROCESSING_RESULT_QUEUE,
    ) -> None:
        self.conn = connection
        self.ch = channel
        self.queue = queue

    def publish(
        self,
        doc_id: str,
        status: str,
        *,
        processing_report: str | None = None,
        extracted_text: str | None = None,
        processing_metrics: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"documentId": doc_id, "processingStatus": status}
        if processing_report is not None:
            payload["processingReport"] = processing_report
        if extracted_text is not None:
            payload["extractedText"] = extracted_text
        if processing_metrics is not None:
            payload["processingMetrics"] = processing_metrics
        body = json.dumps(payload)

        def _do_publish():
            try:
                self.ch.basic_publish(
                    exchange="",
                    routing_key=self.queue,
                    body=body,
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Status publish failed doc_id=%s status=%s: %s", doc_id, status, exc
                )

        self.conn.add_callback_threadsafe(_do_publish)

    def completed(
        self, doc_id: str, extracted_text: str, processing_metrics: str
    ) -> None:
        self.publish(
            doc_id,
            "COMPLETED",
            extracted_text=extracted_text,
            processing_metrics=processing_metrics,
        )

    def failed(
        self,
        doc_id: str,
        message: str,
        report_extra: dict | None = None,
        extracted_text: str | None = None,
        processing_metrics: str | None = None,
    ) -> None:
        report = {"message": message}
        if report_extra:
            report.update(report_extra)
        self.publish(
            doc_id,
            "FAILED",
            processing_report=json.dumps(report),
            extracted_text=extracted_text,
            processing_metrics=processing_metrics,
        )

    def processing(self, doc_id: str) -> None:
        self.publish(doc_id, "PROCESSING")
