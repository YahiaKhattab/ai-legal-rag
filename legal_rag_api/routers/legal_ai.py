from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

from legal_rag.chat_context.contextualizer import QueryContextualizer
from legal_rag.chat_context.memory import ChatMemoryStore
from legal_rag.chat_context.topic_tracker import TopicTracker
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
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationResponse,
    CreateConversationRequest,
    LegalEvidence,
    RenameConversationRequest,
)


router = APIRouter(
    prefix="/legalAi",
    tags=["Legal AI"],
)


@router.post(
    "/Conversation",
    summary="Create a new conversation",
    response_model=ConversationResponse,
)
async def create_conversation(
    request: CreateConversationRequest,
) -> ConversationResponse:
    try:
        settings = Settings()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        conversation = chat_memory.create_conversation(
            title=request.title,
        )

        return ConversationResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    except Exception as exc:
        print("\n========== CREATE CONVERSATION ERROR ==========")
        traceback.print_exc()
        print("==============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to create the conversation.",
        ) from exc


@router.patch(
    "/Conversation/{conversation_id}",
    summary="Rename a conversation",
    response_model=ConversationResponse,
)
async def rename_conversation(
    conversation_id: UUID,
    request: RenameConversationRequest,
) -> ConversationResponse:
    try:
        settings = Settings()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        conversation = chat_memory.rename_conversation(
            conversation_id=conversation_id,
            title=request.title,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return ConversationResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print("\n========== RENAME CONVERSATION ERROR ==========")
        traceback.print_exc()
        print("===============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to rename the conversation.",
        ) from exc


@router.delete(
    "/Conversation/{conversation_id}",
    summary="Delete a conversation",
    status_code=204,
)
async def delete_conversation(
    conversation_id: UUID,
) -> None:
    try:
        settings = Settings()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        deleted = chat_memory.delete_conversation(
            conversation_id=conversation_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

    except HTTPException:
        raise

    except Exception as exc:
        print("\n========== DELETE CONVERSATION ERROR ==========")
        traceback.print_exc()
        print("===============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to delete the conversation.",
        ) from exc


@router.get(
    "/Conversations",
    summary="List conversations",
    response_model=list[ConversationResponse],
)
async def list_conversations() -> list[ConversationResponse]:
    try:
        settings = Settings()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        conversations = chat_memory.list_conversations(
            limit=50,
        )

        return [
            ConversationResponse(
                conversation_id=conversation.conversation_id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            for conversation in conversations
        ]

    except Exception as exc:
        print("\n========== LIST CONVERSATIONS ERROR ==========")
        traceback.print_exc()
        print("==============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to list conversations.",
        ) from exc


@router.get(
    "/Conversation/{conversation_id}",
    summary="Get conversation history",
    response_model=ConversationHistoryResponse,
)
async def get_conversation(
    conversation_id: UUID,
) -> ConversationHistoryResponse:
    try:
        settings = Settings()

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        conversation = chat_memory.get_conversation(
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        messages = chat_memory.get_recent_messages(
            conversation_id=conversation_id,
            limit=1000,
        )

        return ConversationHistoryResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[
                ConversationMessage(
                    message_id=message.message_id,
                    role=message.role,
                    content=message.content,
                    timestamp=message.timestamp,
                )
                for message in messages
            ],
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("\n========== GET CONVERSATION ERROR ==========")
        traceback.print_exc()
        print("============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve the conversation.",
        ) from exc


@router.post(
    "/Ask",
    summary="Answer a legal question",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
) -> AskResponse:
    try:
        settings = Settings()

        conversation_id = request.conversation_id

        chat_memory = ChatMemoryStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        chat_memory.ensure_collection()

        conversation = chat_memory.get_conversation(
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        history = chat_memory.get_recent_messages(
            conversation_id=conversation_id,
            limit=10,
        )

        current_topic = chat_memory.get_current_topic(
            conversation_id=conversation_id,
        )

        topic_tracker_client = OllamaGenerationClient(
            base_url=settings.ollama_url,
            model=settings.contextualization_model,
            timeout_seconds=(
                settings.contextualization_timeout_seconds
            ),
        )

        topic_tracker = TopicTracker(
            client=topic_tracker_client,
        )

        detected_topic = topic_tracker.extract_topic(
            query=request.query,
            previous_topic=current_topic,
            previous_messages=history,
        )

        topic_source_message_id = UUID(
            str(request.conversation_id)
        )

        saved_topic = chat_memory.save_topic(
            conversation_id=conversation_id,
            topic=detected_topic,
            source_message_id=topic_source_message_id,
        )

        contextualizer = QueryContextualizer(
            client=topic_tracker_client,
        )

        contextualized_query = contextualizer.contextualize(
            query=request.query,
            history=history,
            current_topic=saved_topic.topic,
        )

        print("\n========== CHAT CONTEXT ==========")
        print(
            f"[Chat Context] Conversation: "
            f"{conversation_id}"
        )
        print(
            f"[Chat Context] Current Topic: "
            f"{saved_topic.topic}"
        )
        print(
            f"[Chat Context] Original: "
            f"{request.query}"
        )
        print(
            "[Chat Context] Contextualized: "
            f"{contextualized_query}"
        )
        print("==================================")

        pipeline = _build_pipeline(
            settings,
            retrieve_top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
        )

        print("\n========== PIPELINE START ==========")
        print(
            f"[Pipeline] Query: "
            f"{contextualized_query}"
        )

        result: CitedAnswer = pipeline.answer(
            contextualized_query,
            language="mixed",
            filters=RetrievalFilters(),
        )

        print("\n========== PIPELINE RESULT ==========")
        print(
            f"[Pipeline] Answer: "
            f"{result.answer_text.strip()}"
        )
        print(
            f"[Pipeline] Legal excerpts: "
            f"{len(result.legal_excerpts)}"
        )
        print(
            f"[Pipeline] Citations: "
            f"{len(result.citations)}"
        )
        print(
            f"[Pipeline] Full result: "
            f"{result!r}"
        )
        print("=====================================")

        chat_memory.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.query,
        )

        chat_memory.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.answer_text.strip(),
        )

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

        return AskResponse(
            conversation_id=conversation_id,
            question=request.query,
            answer=result.answer_text.strip(),
            selected_legal_evidence=selected_legal_evidence,
            citations=citations,
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("\n========== ASK ERROR ==========")
        print(
            f"[Ask Error] {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        print("================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the legal query.",
        ) from exc


@router.post(
    "/AddNewOpinion",
    summary="Add a new legal opinion document",
)
async def add_new_opinion(
    file: UploadFile = File(...),
) -> dict[str, object]:
    try:
        settings = Settings()

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A file name is required.",
            )

        suffix = Path(file.filename).suffix.lower()

        if suffix not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Supported types are PDF, DOCX, and TXT."
                ),
            )

        with TemporaryDirectory() as temporary_directory:
            temporary_path = (
                Path(temporary_directory)
                / file.filename
            )

            total_bytes = 0

            with temporary_path.open("wb") as output_file:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if (
                        total_bytes
                        > DEFAULT_MAXIMUM_DOCUMENT_BYTES
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "The uploaded file exceeds "
                                "the maximum allowed size."
                            ),
                        )

                    output_file.write(chunk)

            ingestion_pipeline = IngestionPipeline(
                chunking_config=ChunkingConfig(),
            )

            ingestion_result = ingestion_pipeline.ingest(
                temporary_path,
            )

            embedding_config = EmbeddingConfig(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
            )

            encoder = EmbeddingEncoder(
                config=embedding_config,
            )

            batch_embedder = BatchEmbedder(
                encoder=encoder,
            )

            embeddings = batch_embedder.embed(
                ingestion_result.chunks,
            )

            vector_store = QdrantVectorStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection,
            )

            indexer = QdrantIndexer(
                vector_store=vector_store,
            )

            indexed_count = indexer.index(
                chunks=ingestion_result.chunks,
                embeddings=embeddings,
            )

        return {
            "filename": file.filename,
            "indexed_chunks": indexed_count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("\n========== ADD OPINION ERROR ==========")
        traceback.print_exc()
        print("=======================================\n")

        raise HTTPException(
            status_code=500,
            detail="Failed to add the legal opinion.",
        ) from exc  