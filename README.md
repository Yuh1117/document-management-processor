# Document Management Processor

A Python/FastAPI that handles document processing, OCR, semantic search, and AI-powered summarization for the Document Management System.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Supported file types](#supported-file-types)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API endpoints](#api-endpoints)
- [Worker (async processing)](#worker-async-processing)
- [Observability](#observability)
- [MLflow model registration](#mlflow-model-registration)

## Overview

This repository is one of three services that make up the DMS:

| Repository | Role |
|---|---|
| **document-management-be** | Spring Boot REST API — auth, document/folder management, permissions, file storage, RabbitMQ publisher. Also owns the `docker-compose.yml` that runs all three services. |
| **document-management-processor** (this repo) | Python/FastAPI — OCR, chunking, embeddings, Elasticsearch indexing, Gemini summarization, RabbitMQ worker |
| **document-management-fe** | Next.js 16 (App Router) frontend — UI, routing, admin panel, i18n |

This service sits between the Spring Boot backend and Elasticsearch. It:

1. Listens to a RabbitMQ queue for new document events from the backend.
2. Downloads the document from S3, runs OCR (EasyOCR / PyMuPDF) if needed, chunks the text, generates embeddings, and indexes everything into Elasticsearch.
3. Exposes a REST API used by the backend for semantic/full-text/hybrid search and AI-generated summaries.

It runs as **two processes**: the FastAPI server (`app.main`) and the RabbitMQ
worker (`app.worker`). They share the same image and codebase but are started
separately — see [Worker](#worker-async-processing).

## Features

- OCR pipeline with blur/contrast quality validation (EasyOCR + OpenCV)
- Document chunking and embedding generation (Sentence Transformers)
- Redis-backed query embedding cache, with graceful degradation when Redis is unavailable
- Elasticsearch indexing with full-text, semantic, and hybrid search modes
- AI-powered document summarization via Google Gemini, with prompts and model versions tracked in MLflow
- Async document processing via RabbitMQ consumer worker
- Prometheus metrics exported from the worker

## Supported file types

Extraction is dispatched by MIME type (`app/constants/defaults.py`):

| Type | Extraction path |
|---|---|
| PDF | PyMuPDF; falls back to OCR when a page yields fewer than `PDF_MIN_CHARS_PER_PAGE` characters |
| Images (`png`, `jpg`, `jpeg`, `bmp`, `tiff`, `webp`) | EasyOCR, preceded by blur/contrast/dimension validation |
| `docx` / `doc` | python-docx |
| `xlsx` / `xls` | pandas + openpyxl |
| `txt` | Direct read, trying `utf-8`, `utf-8-sig`, `cp1258`, then `latin-1` |

## Tech stack

- Python 3.11+
- FastAPI + Uvicorn
- Elasticsearch 9
- RabbitMQ (pika)
- Redis (query embedding cache)
- EasyOCR, PyMuPDF, python-docx, pandas + openpyxl (document parsing)
- OpenCV headless (image quality validation)
- Sentence Transformers (embeddings)
- Google GenAI SDK (Gemini summarization)
- MLflow (prompt/model registry and experiment tracking)
- prometheus-client (worker metrics)
- AWS S3 (boto3) for document file retrieval

## Prerequisites

- Python 3.11+
- Running Elasticsearch instance
- Running RabbitMQ instance
- Running MLflow tracking server with a registered summarization model
- AWS S3 bucket (or compatible) with document files
- Google Gemini API key (for summarization)
- (Optional) Redis — the embedding cache degrades gracefully without it
- (Optional) GPU for faster EasyOCR — CPU is supported by default

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy and fill in environment variables (see [Configuration](#configuration)).

4. Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Start the RabbitMQ worker (separate process):

```bash
python -m app.worker
```

Or use Docker:

```bash
docker build -t dms-processor .
docker run --env-file .env -p 8000:8000 dms-processor
```

## Configuration

All configuration is via environment variables, read in `app/core/config.py`.
That module is the authoritative list — every variable the service reads is
declared there, together with its default.

Worth knowing without opening the file:

- `PROCESSOR_API_KEY` must match the backend's `dms.processor.api-key`, or every
  request is rejected.
- `REDIS_URL` is optional. If it is unset, or Redis is unreachable, the embedding
  cache is skipped and the service keeps running.
- `OCR_USE_GPU` defaults to `false`; CPU inference works but is much slower.

Other tunable values (chunk size, image thresholds, search scores, etc.) are hardcoded in `app/constants/defaults.py`.

## API endpoints

Every route is guarded by `Depends(verify_api_key)`. Requests must carry the
key matching `PROCESSOR_API_KEY`; in normal operation the backend is the only
caller and supplies it automatically.

| Method   | Path              | Description                                   |
| -------- | ----------------- | --------------------------------------------- |
| `POST`   | `/search`         | Full-text / semantic / hybrid document search |
| `POST`   | `/summarize`      | Generate AI summary for a document            |
| `POST`   | `/models/reload`  | Reload the summarization model from MLflow    |
| `GET`    | `/models`         | List available summarization models           |
| `DELETE` | `/index/{doc_id}` | Delete all indexed chunks for a document      |

API docs are available at `http://localhost:8000/docs` when the server is running.

## Worker (async processing)

The worker (`app/worker.py`) is a long-running RabbitMQ consumer. On each message it:

1. Downloads the document file from S3.
2. Extracts text (PDF via PyMuPDF, images via EasyOCR, DOCX via python-docx).
3. Validates image quality (blur, contrast, dimensions).
4. Chunks the extracted text and generates embeddings.
5. Indexes all chunks into Elasticsearch.

Run it alongside the API server as a separate process or container.

## Observability

The worker starts a Prometheus metrics server on port **8001** (`app/worker.py`).
Metrics are defined in `app/core/metrics.py`:

| Metric | Type | Meaning |
|---|---|---|
| `DOCS_PROCESSED` | Counter | Documents processed by the worker |
| `VALIDATION_FAILURES` | Counter | Documents rejected by quality validation |
| `QUALITY_SCORE` | Histogram | Distribution of computed quality scores |
| `PROCESSING_DURATION` | Histogram | End-to-end processing time per document |

The Prometheus and Grafana configuration lives in the **backend** repository at
`document-management-be/monitoring/`, and is started with its `observability`
compose profile. The FastAPI server does not export metrics — only the worker does.

## MLflow model registration

Summarization prompts and model versions are tracked in MLflow. The summarizer
resolves its model through the `champion` alias
(`models:/$MLFLOW_REGISTERED_MODEL_NAME@champion`), so a registered version
carrying that alias must exist before `/summarize` will work.

```bash
python -m scripts.register_model
```

From the backend repository, the equivalent one-off container is:

```bash
docker compose --profile tools run --rm register
```

`POST /models/reload` makes a running server pick up a newly promoted version
without a restart.
