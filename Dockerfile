# AI Legal RAG service image.
#
# Design decisions (confirmed with the requester):
#   - Embedding + reranker models are baked into the image at build time
#     (instant startup, no internet needed at runtime for those two models).
#   - OCR (paddleocr) is included, since scanned PDFs will be ingested.
#   - Ollama itself is NOT baked in here; it runs as its own container
#     (see docker-compose.yml) and this service talks to it over HTTP.
#
# NOTE on OCR models specifically: PaddleOCR/PaddleX has a documented issue
# in some 3.x releases where it still tries to reach its model-hosting
# platforms at first call even when a local cache exists, and fails hard if
# there is no outbound network at that moment. We pre-download the OCR
# models below to avoid this in the common case, but if the deployment
# environment is fully air-gapped, test the /legalAi/AddNewOpinion endpoint
# with a real scanned PDF before relying on it - if it fails, this container
# will need outbound HTTPS access allowed (at least on first OCR call).

FROM python:3.11-slim

# System libraries required by opencv-python (a transitive paddleocr
# dependency) and curl for the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. HOME is set explicitly so that both the
# HuggingFace cache (~/.cache/huggingface) and the PaddleX/PaddleOCR cache
# (~/.paddlex) land in a directory this user actually owns.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser legal_rag_api ./legal_rag_api

USER appuser

# Install the package with the OCR extra (paddleocr + paddlepaddle).
RUN pip install --no-cache-dir --user ".[ocr]"

# Pre-download models at build time so the container starts instantly and
# does not depend on internet access at runtime for embedding/reranking.
# This is the same model-loading code the app uses at runtime, so if this
# step succeeds the models are guaranteed to be usable offline afterwards.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-base'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# Pre-download the Arabic + English OCR models (see the note at the top of
# this file about PaddleOCR's network-on-first-call behavior).
RUN python -c "\
from legal_rag.ingestion.ocr import PaddleOcrEngine; \
PaddleOcrEngine(language='ar'); \
PaddleOcrEngine(language='en')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# NOTE: this is a single-process server. If the backend team needs more
# throughput, ask them to run multiple replicas of this container behind a
# load balancer rather than adding uvicorn --workers, since the embedding/
# reranker/OCR models are loaded per-process and multiple workers would
# multiply memory usage.
CMD ["uvicorn", "legal_rag_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
