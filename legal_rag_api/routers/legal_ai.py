"""HTTP endpoints for legal question answering and opinion ingestion."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile

from legal_rag.config import Settings
from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.embeddings.models import EmbeddingConfig
from legal_rag.ingestion.models import ChunkingConfig
from legal_rag.ingestion.pipeline import IngestionPipeline
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_DOCUMENT_BYTES
from legal_rag.query.cli import _build_pipeline
from legal_rag.query.models import CitedAnswer
from legal_rag.query.retriever import RetrievalFilters
from legal_rag.vector_store.indexer import QdrantIndexer
from legal_rag.vector_store.qdrant import QdrantVectorStore

from legal_rag_api.schemas import AskRequest, AskResponse


router = APIRouter(
    prefix="/legalAi",
    tags=["Legal AI"],
)


@router.post(
    "/Ask",
    summary="Answer a legal question",
    response_model=AskResponse,
)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a legal question using the configured RAG pipeline."""

    try:
        settings = Settings()

        pipeline = _build_pipeline(
            settings,
            retrieve_top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
        )

        result: CitedAnswer = pipeline.answer(
            request.query,
            language="mixed",
            filters=RetrievalFilters(),
        )

        source = ""

        if result.legal_excerpts:
            source = (
                result.legal_excerpts[0].source_file
                or result.legal_excerpts[0].chunk_id
            )

        return AskResponse(
            answer=result.answer_text,
            source=source,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the legal query.",
        ) from exc


def _build_indexer(settings: Settings) -> QdrantIndexer:
    """Build the configured Qdrant indexer."""

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )

    embedder = BatchEmbedder(
        EmbeddingEncoder(
            EmbeddingConfig(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
            )
        )
    )

    return QdrantIndexer(
        store=store,
        embedder=embedder,
    )


@router.post(
    "/AddNewOpinion",
    summary="Add a new legal opinion",
)
async def add_new_opinion(
    file: UploadFile = File(...),
) -> str:
    """Ingest and index one legal opinion into Qdrant."""

    if not file.filename:
        return "Failed To Add"

    try:
        settings = Settings()

        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / file.filename

            file_content = await file.read()
            temporary_path.write_bytes(file_content)

            ingestion_pipeline = IngestionPipeline(
                expected_language="auto",
                chunking=ChunkingConfig(
                    target_tokens=400,
                    overlap_tokens=60,
                    maximum_tokens=480,
                ),
                maximum_document_bytes=DEFAULT_MAXIMUM_DOCUMENT_BYTES,
            )

            summary = ingestion_pipeline.ingest(
                temporary_path,
                document_version=1,
                document_type="unknown",
                source="unknown",
            )

            if summary.chunks == 0:
                return "Failed To Add"

            indexer = _build_indexer(settings)
            indexer.ensure_collection()
            indexer.index_file(summary.chunks_output)

        return "Added To Database Successfully"

    except Exception:
        return "Failed To Add"