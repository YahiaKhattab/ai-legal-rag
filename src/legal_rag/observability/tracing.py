"""Owned OTel provider with no-op defaults and an explicit metadata allowlist.

No function arguments or exception messages are serialized. Dependencies are
imported only when enabled. This module never installs a global tracer provider.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from functools import wraps
from inspect import signature
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from legal_rag.observability.config import ObservabilitySettings

P = ParamSpec("P")
R = TypeVar("R")
_logger = logging.getLogger(__name__)
_lock = Lock()
_provider: Any = None
_tracer: Any = None
_session_salt = secrets.token_bytes(32)

# Explicit scalar keys only. No URLs, text, session IDs, exception messages or bodies.
_ALLOWED = frozenset(
    {
        "session.id",
        "rag.sufficient",
        "rag.reason",
        "rag.language",
        "rag.candidate_count",
        "rag.used_chunk_count",
        "rag.source_count",
        "rag.top_dense_score",
        "rag.top_rerank_score",
        "rag.exact_identifier_match",
        "rag.prompt_version",
        "rag.result_count",
        "rag.history_count",
        "rag.query_changed",
        "rag.input_characters",
        "rag.output_characters",
        "rag.attempt",
        "rag.validation",
        "rag.generation_purpose",
        "rag.exception_type",
        "rag.minimum_dense_score",
        "rag.minimum_rerank_score",
        "rag.identifier_override_score",
        "rag.minimum_lexical_overlap",
        "rag.gate_enabled",
        "rag.retrieve_top_k",
        "rag.rerank_top_n",
        "rag.evidence_top_n",
        "rag.context_limit",
        "rag.structured_output",
        "llm.model_name",
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "rag.embedding_dimensions",
        "rag.embedding_model",
        "rag.rerank_model",
        "rag.dense_scores",
        "rag.raw_rerank_scores",
        "rag.adjusted_rerank_scores",
    }
)


def initialize(settings: ObservabilitySettings | None = None) -> bool:
    """Initialize once per process. Unavailable telemetry does not stop the app."""
    global _provider, _tracer
    with _lock:
        if _provider is not None:
            return True
        try:
            if settings is None:
                from legal_rag.observability.config import ObservabilitySettings

                settings = ObservabilitySettings()
            if not settings.enabled:
                return False
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ALWAYS_ON

            # Explicit Resource avoids host/process metadata from automatic detectors.
            provider = TracerProvider(
                sampler=ALWAYS_ON,
                resource=Resource(
                    {"service.name": "legal-rag", "openinference.project.name": settings.project}
                ),
                shutdown_on_exit=False,
            )
            exporter = OTLPSpanExporter(endpoint=settings.endpoint, timeout=10)
            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=512,
                    max_export_batch_size=64,
                    schedule_delay_millis=500,
                    export_timeout_millis=2000,
                )
            )
            _tracer = provider.get_tracer("legal_rag.manual", "0.1.0")
            _provider = provider
            return True
        except Exception:
            _logger.warning("Phoenix tracing unavailable; continuing without telemetry.")
            return False


def shutdown() -> None:
    """Drain on a daemon thread; application shutdown waits at most three seconds."""
    global _provider, _tracer
    with _lock:
        provider, _provider, _tracer = _provider, None, None
    if provider is None:
        return

    def drain() -> None:
        try:
            provider.force_flush(timeout_millis=5000)
            provider.shutdown()
        except Exception:
            _logger.warning("Phoenix tracing shutdown failed.")

    worker = Thread(target=drain, daemon=True, name="rag-trace-shutdown")
    worker.start()
    worker.join(timeout=3)
    if worker.is_alive():
        _logger.warning("Phoenix tracing shutdown timed out; pending spans may be lost.")


def attributes(**values: object) -> None:
    """Attach only approved, bounded metadata to the current span."""
    if _tracer is None:
        return
    try:
        from opentelemetry.trace import get_current_span

        current = get_current_span()
        for key, value in values.items():
            if key not in _ALLOWED or value is None:
                continue
            if isinstance(value, (bool, int, float)):
                current.set_attribute(key, value)
            elif isinstance(value, str):
                current.set_attribute(key, value[:120])
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                current.set_attribute(key, value[:50])
    except Exception:
        # Instrumentation must never replace an application result or exception.
        pass


def correlate_session(session_id: object) -> None:
    """Pseudonymous session correlation within this worker's lifetime only."""
    if _tracer is None:
        return
    digest = hmac.new(_session_salt, str(session_id).encode("utf-8"), hashlib.sha256).hexdigest()
    attributes(**{"session.id": digest})


@contextmanager
def span(name: str, kind: str = "CHAIN") -> Iterator[None]:
    tracer = _tracer
    if tracer is None:
        yield
        return
    try:
        manager = tracer.start_as_current_span(
            name,
            attributes={"openinference.span.kind": kind},
            record_exception=False,
            set_status_on_exception=False,
        )
        current = manager.__enter__()
    except Exception:
        yield
        return
    try:
        yield
    except BaseException as exc:
        try:
            from opentelemetry.trace import StatusCode

            current.set_attribute("rag.exception_type", type(exc).__name__)
            current.set_status(StatusCode.ERROR)
        except Exception:
            pass
        raise
    finally:
        with suppress(Exception):
            manager.__exit__(None, None, None)


def _result_metadata(name: str, result: object) -> None:
    """Read known output fields, never repr()/asdict() arbitrary application data."""
    if _tracer is None:
        return
    try:
        if isinstance(result, list):
            attributes(**{"rag.result_count": len(result)})
        if name == "rag.answer":
            attributes(
                **{
                    "rag.language": getattr(result, "language", None),
                    "rag.prompt_version": getattr(result, "prompt_version", None),
                }
            )
            result = getattr(result, "retrieval", None)
        if name in {"rag.answer", "evidence.assess"}:
            attributes(
                **{
                    "rag." + key: getattr(result, key, None)
                    for key in (
                        "sufficient",
                        "reason",
                        "candidate_count",
                        "used_chunk_count",
                        "source_count",
                        "top_dense_score",
                        "top_rerank_score",
                        "exact_identifier_match",
                    )
                }
            )
    except Exception:
        pass


def traced(name: str, kind: str = "CHAIN") -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with span(name, kind):
                result = function(*args, **kwargs)
                _result_metadata(name, result)
                return result

        return wrapped

    return decorate


def traced_async(
    name: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with span(name):
                return await function(*args, **kwargs)

        # FastAPI must resolve endpoint annotations in the original module.
        wrapped.__signature__ = signature(function, eval_str=True)  # type: ignore[attr-defined]
        return wrapped

    return decorate


def tracing_process(function: Callable[P, R]) -> Callable[P, R]:
    """CLI lifecycle; the API uses lifespan instead."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        initialize()
        try:
            with span("cli.query"):
                return function(*args, **kwargs)
        finally:
            shutdown()

    return wrapped
