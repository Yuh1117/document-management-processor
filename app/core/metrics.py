from prometheus_client import Counter, Histogram

DOCS_PROCESSED = Counter(
    "documents_processed_total",
    "Total documents processed by the Processing pipeline",
    ["status"],
)

QUALITY_SCORE = Histogram(
    "text_quality_score",
    "Distribution of quality scores (0-100)",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

VALIDATION_FAILURES = Counter(
    "validation_failures_total",
    "Total data-validation failures before processing",
)

PROCESSING_DURATION = Histogram(
    "processing_duration_seconds",
    "End-to-end document processing time",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
