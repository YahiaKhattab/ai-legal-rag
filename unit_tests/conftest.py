"""Shared pytest fixtures and import path setup.

This test suite lives outside the project's own ``tests/`` folder so it
never overwrites or conflicts with work already written by teammates.
It imports the same ``legal_rag`` package straight from ``src/`` using the
``pythonpath`` setting in pytest.ini (see pytest.ini next to this file).
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_stub_if_missing(module_name: str, attributes: dict[str, object]) -> None:
    """Make heavy ML packages importable without actually installing them.

    sentence-transformers (needs torch) and paddleocr (needs paddlepaddle)
    are multi-GB dependencies that are completely unnecessary for unit
    testing: every test that touches code using them patches the specific
    class it needs (SentenceTransformer, CrossEncoder, PaddleOCR) with a
    fake object. But Python still needs `import sentence_transformers` to
    succeed at module-load time for files like embeddings/encoder.py,
    query/query_embedder.py, and query/reranker.py to be importable at all.

    If the real package IS installed (e.g. in a full project dev
    environment), this function does nothing and the real package wins.
    """

    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except ImportError:
        pass

    stub = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(stub, name, value)
    sys.modules[module_name] = stub


class _UnusableStub:
    """Placeholder class. Tests must patch this out; instantiating it
    directly means a test forgot to mock the real dependency."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is a test-environment stub with no real "
            "implementation. Patch it (unittest.mock.patch / monkeypatch) "
            "before use instead of instantiating it directly."
        )


_install_stub_if_missing(
    "sentence_transformers",
    {
        "SentenceTransformer": type("SentenceTransformer", (_UnusableStub,), {}),
        "CrossEncoder": type("CrossEncoder", (_UnusableStub,), {}),
    },
)
_install_stub_if_missing(
    "paddleocr",
    {"PaddleOCR": type("PaddleOCR", (_UnusableStub,), {})},
)
