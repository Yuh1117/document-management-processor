import logging

import pika
from prometheus_client import start_http_server as start_metrics_server

from app.core.config import (
    ELASTICSEARCH_INDEX,
    RABBITMQ_DOCUMENT_QUEUE,
    RABBITMQ_PROCESSING_RESULT_QUEUE,
    RABBITMQ_URL,
)
from app.core.metrics import (
    DOCS_PROCESSED,
)
from app.embedding.service import embedding_service
from app.extraction.service import extraction_service
from app.extraction.text_quality import text_quality_service
from app.worker.constants import (
    METRICS_PORT,
    RABBITMQ_BLOCKED_CONNECTION_TIMEOUT,
    RABBITMQ_HEARTBEAT,
    TORCH_NUM_THREADS,
    WORKER_THREADS,
)
from app.worker.processor import DocumentProcessor

logger = logging.getLogger(__name__)


def _limit_cpu_threads() -> None:
    import cv2
    import torch

    torch.set_num_threads(TORCH_NUM_THREADS)
    # OpenCV (used for OCR image checks) spawns its own pool per call.
    cv2.setNumThreads(1)


def main() -> None:
    _limit_cpu_threads()
    start_metrics_server(METRICS_PORT)
    logger.info("Prometheus metrics server started on :%s", METRICS_PORT)
    for status in ("completed", "failed"):
        DOCS_PROCESSED.labels(status=status)
    logger.info(
        "Worker starting queue=%s index=%s",
        RABBITMQ_DOCUMENT_QUEUE,
        ELASTICSEARCH_INDEX,
    )

    processor = DocumentProcessor(
        embedding_service, extraction_service, text_quality_service
    )

    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = RABBITMQ_HEARTBEAT
    params.blocked_connection_timeout = RABBITMQ_BLOCKED_CONNECTION_TIMEOUT
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_DOCUMENT_QUEUE, durable=True)
    channel.queue_declare(queue=RABBITMQ_PROCESSING_RESULT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=WORKER_THREADS)
    channel.basic_consume(
        queue=RABBITMQ_DOCUMENT_QUEUE,
        on_message_callback=lambda ch, method, props, body: processor.process_message(
            connection, ch, method, props, body
        ),
    )

    logger.info(
        "Consuming messages on %s (prefetch=%s, threads=%s)",
        RABBITMQ_DOCUMENT_QUEUE,
        WORKER_THREADS,
        WORKER_THREADS,
    )
    channel.start_consuming()


if __name__ == "__main__":
    main()
