from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from legal_rag.chat_context.contextualizer import QueryContextualizer
from legal_rag.chat_context.memory import ChatMemoryStore
from legal_rag.config import Settings
from legal_rag.embeddings.batch import BatchEmbedder
from legal_rag.embeddings.encoder import EmbeddingEncoder
from legal_rag.embeddings.models import EmbeddingConfig
from legal_rag.ingestion.models import ChunkingConfig
from legal_rag.ingestion.pipeline import IngestionPipeline
from legal_rag.ingestion.validation import DEFAULT_MAXIMUM_DOCUMENT_BYTES
from legal_rag.query.cli import _build_pipeline
from legal_rag.query.models import CitedAnswer
from legal_rag.query.ollama_client import OllamaGenerationClient
from legal_rag.query.retriever import RetrievalFilters
from legal_rag.vector_store.indexer import QdrantIndexer
from legal_rag.vector_store.qdrant import QdrantVectorStore

from legal_rag_api.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    LegalEvidence,
)


router = APIRouter(
    prefix="/legalAi",
    tags=["Legal AI"],
)


# ======================================================================
# ASK
# ======================================================================


@router.post(
    "/Ask",
    summary="Answer a legal question",
    response_model=AskResponse,
)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a legal question using chat context and the legal RAG pipeline."""

    try:
        settings = Settings()

        # ==============================================================
        # CHAT SESSION
        # ==============================================================

        # Generate a new internal session ID when this is a new
        # conversation. The user does not need to provide one manually.
        session_id = request.session_id or uuid4()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        # Retrieve recent conversation history for this session.
        # Chat history is context only and is NEVER treated as legal
        # evidence.
        history = chat_memory.get_recent_messages(
            session_id=session_id,
            limit=10,
        )

        # ==============================================================
        # QUERY CONTEXTUALIZATION
        # ==============================================================

        # Use the dedicated non-thinking model for contextualization.
        # qwen3:4b remains responsible for the final legal answer.
        contextualization_client = OllamaGenerationClient(
            base_url=settings.ollama_url,
            model=settings.contextualization_model,
            timeout_seconds=settings.contextualization_timeout_seconds,
        )

        contextualizer = QueryContextualizer(
            client=contextualization_client,
        )

        # For a new conversation, this returns the original query
        # unchanged. For a follow-up question, it rewrites the query
        # into a standalone legal retrieval query.
        contextualized_query = contextualizer.contextualize(
            query=request.query,
            history=history,
        )
        print(f"[Chat Context] Original: {request.query}")
        print(f"[Chat Context] Contextualized: {contextualized_query}")

        # ==============================================================
        # CURRENT RAG PIPELINE
        # ==============================================================

        pipeline = _build_pipeline(
            settings,
            retrieve_top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
        )

        # IMPORTANT:
        # Only the contextualized query enters legal retrieval.
        # Chat history itself is never sent to the legal retriever.
        result: CitedAnswer = pipeline.answer(
            contextualized_query,
            language="mixed",
            filters=RetrievalFilters(),
        )

        # ==============================================================
        # SAVE CONVERSATION
        # ==============================================================

        # Save the ORIGINAL user question so the conversation remains
        # faithful to what the user actually asked.
        chat_memory.save_message(
            session_id=session_id,
            role="user",
            content=request.query,
        )

        # Save the final assistant answer.
        chat_memory.save_message(
            session_id=session_id,
            role="assistant",
            content=result.answer_text.strip(),
        )

        # ==============================================================
        # SELECTED LEGAL EVIDENCE
        # ==============================================================

        selected_legal_evidence: list[LegalEvidence] = []

        for index, excerpt in enumerate(
            result.legal_excerpts,
            start=1,
        ):
            selected_legal_evidence.append(
                LegalEvidence(
                    citation=f"[{index}]",
                    source=(
                        excerpt.source_file
                        or excerpt.chunk_id
                    ),
                    page=excerpt.page,
                    section=excerpt.section_title,
                    evidence=excerpt.text.strip(),
                )
            )

        # ==============================================================
        # CITATIONS
        # ==============================================================

        citations: list[CitationResponse] = []

        for citation in result.citations:
            citations.append(
                CitationResponse(
                    marker=citation.marker,
                    source=(
                        citation.source_file
                        or citation.document_id
                    ),
                    section=citation.section_title,
                    page=citation.page,
                )
            )

        # ==============================================================
        # USER-FACING RESPONSE
        # ==============================================================

        return AskResponse(
            session_id=session_id,
            question=request.query,
            answer=result.answer_text.strip(),
            selected_legal_evidence=selected_legal_evidence,
            citations=citations,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the legal query.",
        ) from exc


# ======================================================================
# QDRANT INDEXER
# ======================================================================


def _build_indexer(settings: Settings) -> QdrantIndexer:
    """Build the configured Qdrant indexer."""

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        omit_normalized_text=settings.qdrant_omit_normalized_text,
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


# ======================================================================
# ADD NEW OPINION
# ======================================================================


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

            temporary_path = (
                Path(temporary_directory)
                / file.filename
            )

            file_content = await file.read()

            if not file_content:
                return "Failed To Add"

            temporary_path.write_bytes(
                file_content
            )

            # ==========================================================
            # INGESTION
            # ==========================================================

            ingestion_pipeline = IngestionPipeline(
                expected_language="auto",
                chunking=ChunkingConfig(
                    target_tokens=400,
                    overlap_tokens=60,
                    maximum_tokens=480,
                ),
                maximum_document_bytes=(
                    DEFAULT_MAXIMUM_DOCUMENT_BYTES
                ),
            )

            summary = ingestion_pipeline.ingest(
                temporary_path,
                Path(temporary_directory),
                document_version=1,
                document_type="unknown",
                source="unknown",
            )

            if summary.chunks == 0:
                return "Failed To Add"

            # ==========================================================
            # INDEX INTO QDRANT
            # ==========================================================

            indexer = _build_indexer(settings)

            indexer.ensure_collection()

            indexer.index_file(
                summary.chunks_output
            )

        return "Added To Database Successfully"

    except Exception:
        return "Failed To Add"