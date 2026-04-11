# Document Management Processor

A Python/FastAPI that handles document processing, OCR, semantic search, and AI-powered summarization for the Document Management System.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API endpoints](#api-endpoints)
- [Worker (async processing)](#worker-async-processing)

## Overview

This service sits between the Spring Boot backend and Elasticsearch. It:

1. Listens to a RabbitMQ queue for new document events from the backend.
2. Downloads the document from S3, runs OCR (EasyOCR / PyMuPDF) if needed, chunks the text, generates embeddings, and indexes everything into Elasticsearch.
3. Exposes a REST API used by the backend for semantic/full-text/hybrid search and AI-generated summaries.

## Features

- OCR pipeline with blur/contrast quality validation (EasyOCR + OpenCV)
- Document chunking and embedding generation (Sentence Transformers)
- Elasticsearch indexing with full-text, semantic, and hybrid search modes
- AI-powered document summarization via Google Gemini
- Prometheus metrics exposed at `/metrics`
- Async document processing via RabbitMQ consumer worker

## Tech stack

- Python 3.11+
- FastAPI + Uvicorn
- Elasticsearch 9
- RabbitMQ (pika)
- EasyOCR, PyMuPDF, python-docx (document parsing)
- OpenCV (image quality validation)
- Sentence Transformers (embeddings)
- Google Generative AI SDK (Gemini summarization)
- MLflow (experiment tracking)
- Prometheus + prometheus-fastapi-instrumentator (metrics)
- AWS S3 (boto3) for document file retrieval

## Prerequisites

- Python 3.11+
- Running Elasticsearch instance
- Running RabbitMQ instance
- AWS S3 bucket (or compatible) with document files
- Google Gemini API key (for summarization)
- (Optional) GPU for faster OCR — CPU works fine with `OCR_USE_GPU=false`

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

All configuration is via environment variables:

| Variable | Description |
|---|---|
| `RABBITMQ_URL` | RabbitMQ connection URL |
| `RABBITMQ_DOCUMENT_QUEUE` | Queue name for document events |
| `ELASTICSEARCH_HOST` | Elasticsearch host URL |
| `ELASTICSEARCH_INDEX` | Index name for documents |
| `BACKEND_BASE_URL` | Base URL of the Spring Boot backend |
| `AWS_S3_ACCESS_KEY` | S3 access key |
| `AWS_S3_SECRET_KEY` | S3 secret key |
| `AWS_S3_REGION` | S3 region |
| `SENTENCE_TRANSFORMER_MODEL_NAME` | HuggingFace model name for embeddings |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL_NAME` | Gemini model name (e.g. `gemini-2.0-flash`) |
| `SUMMARIZE_PROMPT_VERSION` | MLflow prompt version tag |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `MLFLOW_SUMMARIZE_MODEL_URI` | MLflow model URI for summarization |
| `CHUNK_SIZE` | Token chunk size for indexing (default: `250`) |
| `CHUNK_OVERLAP` | Token overlap between chunks (default: `50`) |
| `OCR_USE_GPU` | Use GPU for EasyOCR (default: `false`) |
| `OCR_MIN_QUALITY_SCORE` | Minimum OCR quality score (default: `60`) |
| `LAPLACIAN_VAR_THRESHOLD` | Blur detection threshold (default: `100.0`) |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/search` | Full-text / semantic / hybrid document search |
| `POST` | `/summarize` | Generate AI summary for a document |
| `POST` | `/indexing` | Manually trigger document indexing |
| `GET` | `/metrics` | Prometheus metrics |

API docs are available at `http://localhost:8000/docs` when the server is running.

## Worker (async processing)

The worker (`app/worker.py`) is a long-running RabbitMQ consumer. On each message it:

1. Downloads the document file from S3.
2. Extracts text (PDF via PyMuPDF, images via EasyOCR, DOCX via python-docx).
3. Validates image quality (blur, contrast, dimensions).
4. Chunks the extracted text and generates embeddings.
5. Indexes all chunks into Elasticsearch.
6. Reports processing metrics (duration, quality score, validation failures) to Prometheus.

Run it alongside the API server as a separate process or container.
