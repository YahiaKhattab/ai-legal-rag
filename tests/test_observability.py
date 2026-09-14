"""Metadata-only tracing tests; no Phoenix, Qdrant or model service required."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import sys
import types
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from legal_rag.observability import tracing
from legal_rag.observability.config import ObservabilitySettings


@pytest.fixture(autouse=True)
def lightweight_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Heavy packages are not needed: this path uses injected retrieval/model fakes.
    if (
        "sentence_transformers" not in sys.modules
        and importlib.util.find_spec("sentence_transformers") is None
    ):
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = Mock()  # type: ignore[attr-defined]
        stub.CrossEncoder = Mock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({}), sampler=ALWAYS_ON, shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def test_disabled_never_imports_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("opentelemetry"):
            raise AssertionError("disabled tracing imported OTel")
        return original(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert not tracing.initialize(ObservabilitySettings.model_construct(enabled=False))
    with tracing.span("disabled"):
        tracing.attributes(**{"rag.reason": "sufficient"})
    tracing.shutdown()


def test_parenting_privacy_and_original_exception(exported: InMemorySpanExporter) -> None:
    error = ValueError("SECRET exception body")
    with tracing.span("root"), pytest.raises(ValueError) as raised, tracing.span("child", "LLM"):
        tracing.attributes(**{"rag.reason": "sufficient", "prompt": "SECRET question"})
        raise error
    spans = {s.name: s for s in exported.get_finished_spans()}
    assert raised.value is error
    assert spans["child"].parent is not None
    assert spans["child"].parent.span_id == spans["root"].context.span_id
    assert spans["child"].status.is_ok is False
    assert (spans["child"].attributes or {})["rag.exception_type"] == "ValueError"
    assert not spans["child"].events
    assert "SECRET" not in "".join(s.to_json() for s in spans.values())


def test_async_roots_do_not_leak_context(exported: InMemorySpanExporter) -> None:
    @tracing.traced_async("request")
    async def request() -> None:
        await asyncio.sleep(0)
        with tracing.span("child"):
            await asyncio.sleep(0)

    async def run() -> None:
        await asyncio.gather(request(), request())

    asyncio.run(run())
    spans = exported.get_finished_spans()
    roots = [s for s in spans if s.name == "request"]
    assert len({s.context.trace_id for s in roots}) == 2
    assert all(s.parent is None for s in roots)
    assert {s.parent.span_id for s in spans if s.name == "child" and s.parent is not None} == {
        s.context.span_id for s in roots
    }


def test_initialization_once_and_export_failure_does_not_change_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    exporter = Mock()
    exporter.export.side_effect = RuntimeError("unavailable")
    factory = Mock(return_value=exporter)
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", factory)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_tracer", None)
    config = ObservabilitySettings.model_construct(enabled=True)
    assert tracing.initialize(config)
    assert tracing.initialize(config)
    with tracing.span("answer"):
        result = "unchanged"
    tracing.shutdown()
    assert result == "unchanged"
    assert factory.call_count == 1
    exporter.export.assert_called_once()


def test_missing_optional_dependency_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("opentelemetry"):
            raise ImportError("missing extra")
        return original(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(builtins, "__import__", missing)
    assert not tracing.initialize(ObservabilitySettings.model_construct(enabled=True))


def test_chat_rewrite_and_ollama_export_no_content(
    exported: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from legal_rag.chat_context.contextualizer import QueryContextualizer
    from legal_rag.chat_context.models import ChatMessage
    from legal_rag.query.ollama_client import OllamaGenerationClient

    def post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "message": {"content": "PRIVATE rewritten question"},
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    client = OllamaGenerationClient()
    contextualizer = QueryContextualizer(client)
    with tracing.span("api.ask"):
        assert contextualizer.contextualize("PRIVATE first", []) == "PRIVATE first"
        history = [
            ChatMessage(
                session_id=uuid4(),
                message_id=uuid4(),
                role="user",
                content="PRIVATE history",
                timestamp=datetime.now(UTC),
            )
        ]
        assert (
            contextualizer.contextualize("PRIVATE followup", history)
            == "PRIVATE rewritten question"
        )
    spans = exported.get_finished_spans()
    calls = [s for s in spans if s.name == "generation.attempt"]
    assert len(calls) == 1
    assert (calls[0].attributes or {})["rag.generation_purpose"] == "contextualization"
    assert (calls[0].attributes or {})["llm.token_count.prompt"] == 12
    assert "PRIVATE" not in "".join(s.to_json() for s in spans)


def test_real_gate_rejection_prevents_final_generation(
    exported: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from legal_rag.query.evidence_sufficiency import EvidenceSufficiencyEvaluator
    from legal_rag.query.pipeline import RAGAnswerPipeline

    generator = Mock()
    pipeline = RAGAnswerPipeline(
        retriever=Mock(search=Mock(return_value=[])),
        reranker=Mock(rerank=Mock(return_value=[])),
        generator=generator,
        sufficiency_evaluator=EvidenceSufficiencyEvaluator(),
    )
    result = pipeline.answer("PRIVATE unsupported question", language="en")
    assert result.retrieval is not None
    assert result.retrieval.sufficient is False
    generator.generate.assert_not_called()
    spans = exported.get_finished_spans()
    assert "evidence.assess" in {s.name for s in spans}
    assert "generation.attempt" not in {s.name for s in spans}
    gate = next(s for s in spans if s.name == "evidence.assess")
    assert (gate.attributes or {})["rag.minimum_dense_score"] == 0.855
    serialized = "".join(s.to_json() for s in spans)
    assert "PRIVATE" not in serialized
    monkeypatch.setattr(tracing, "_tracer", None)
    assert pipeline.answer("PRIVATE unsupported question", language="en") == result


def test_api_schema_and_request_trace(
    exported: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from legal_rag_api.main import app
    from legal_rag_api.routers import legal_ai

    from legal_rag.query.evidence_sufficiency import EvidenceSufficiencyEvaluator
    from legal_rag.query.pipeline import RAGAnswerPipeline

    pipeline = RAGAnswerPipeline(
        retriever=Mock(search=Mock(return_value=[])),
        reranker=Mock(rerank=Mock(return_value=[])),
        generator=Mock(),
        sufficiency_evaluator=EvidenceSufficiencyEvaluator(),
    )
    memory = Mock()
    memory.get_recent_messages.return_value = []
    monkeypatch.setattr(legal_ai, "ChatMemoryStore", Mock(return_value=memory))
    monkeypatch.setattr(legal_ai, "_build_pipeline", Mock(return_value=pipeline))
    # The exporter fixture owns provider lifecycle for this test.
    with TestClient(app) as client:
        response = client.post("/legalAi/Ask", json={"query": "PRIVATE question"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["question"] == "PRIVATE question"
        assert payload["session_id"]
        assert payload["citations"] == []
        spans = exported.get_finished_spans()
    roots = [s for s in spans if s.name == "api.ask"]
    assert len(roots) == 1
    assert (roots[0].attributes or {}).get("session.id")
    assert payload["session_id"] not in "".join(s.to_json() for s in spans)
    rag = next(s for s in spans if s.name == "rag.answer")
    assert rag.parent is not None
    assert rag.parent.span_id == roots[0].context.span_id
    assert memory.save_message.call_count == 2
    assert "PRIVATE" not in "".join(s.to_json() for s in spans)


def test_supported_answer_retry_parity_and_attempt_spans(
    exported: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from legal_rag.query.evidence_sufficiency import EvidenceSufficiencyEvaluator
    from legal_rag.query.models import RerankedChunk
    from legal_rag.query.ollama_client import OllamaGenerationClient
    from legal_rag.query.pipeline import RAGAnswerPipeline

    chunk = RerankedChunk(
        chunk_id="synthetic",
        document_id="synthetic",
        score=0.95,
        rerank_score=8.0,
        text="Article 3\nClear contract terms are required.",
        section_title="Article 3",
        source_file="PRIVATE file.txt",
    )
    responses = iter(
        [
            "PRIVATE invalid JSON",
            '{"answer":"Clear contract terms are required.",'
            '"evidence_ids":["E1"],"insufficient_evidence":false}',
        ]
        * 2
    )

    def post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "message": {"content": next(responses)},
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    pipeline = RAGAnswerPipeline(
        retriever=Mock(search=Mock(return_value=[chunk])),
        reranker=Mock(rerank=Mock(return_value=[chunk])),
        generator=OllamaGenerationClient(),
        sufficiency_evaluator=EvidenceSufficiencyEvaluator(),
    )
    enabled = pipeline.answer("What does article 3 require?", language="en")
    assert enabled.answer_text == "Clear contract terms are required. [1]"
    spans = exported.get_finished_spans()
    attempts = [s for s in spans if s.name == "answer.attempt"]
    assert [(s.attributes or {})["rag.attempt"] for s in attempts] == [1, 2]
    calls = [s for s in spans if s.name == "generation.attempt"]
    assert len(calls) == 2
    assert {s.parent.span_id for s in calls if s.parent is not None} == {
        s.context.span_id for s in attempts
    }
    assert "PRIVATE" not in "".join(s.to_json() for s in spans)
    monkeypatch.setattr(tracing, "_tracer", None)
    assert pipeline.answer("What does article 3 require?", language="en") == enabled
