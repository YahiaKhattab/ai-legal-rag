"""Unit tests for legal_rag.ingestion.tokenization.E5TokenCounter.

E5TokenCounter.load() can either download a real tokenizer from the
internet or read a local tokenizer.json file. A unit test must not hit
the network, so we build a tiny fake tokenizer object (matching the
interface the code actually uses: `.encode(text).ids`) and inject it
through the class constructor directly, and separately test `.load()`'s
error handling by making the import itself fail.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from legal_rag.ingestion.tokenization import E5TokenCounter, E5_PASSAGE_PREFIX


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """Splits on whitespace and returns one fake token id per word."""

    def encode(self, text, add_special_tokens=True):
        words = text.split()
        return _FakeEncoding(ids=list(range(len(words))))


def test_count_passage_adds_the_e5_prefix_before_counting():
    tokenizer = _FakeTokenizer()
    counter = E5TokenCounter(tokenizer, name="fake")

    # "passage: one two three" -> 4 words -> 4 fake tokens.
    assert counter.count_passage("one two three") == 4


def test_count_content_does_not_add_the_prefix():
    tokenizer = _FakeTokenizer()
    counter = E5TokenCounter(tokenizer, name="fake")

    assert counter.count_content("one two three") == 3


def test_name_property_returns_configured_name():
    counter = E5TokenCounter(_FakeTokenizer(), name="my-tokenizer@rev")
    assert counter.name == "my-tokenizer@rev"


def test_load_from_local_tokenizer_file(tmp_path, monkeypatch):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    fake_tokenizer_class = type(
        "FakeTokenizerClass",
        (),
        {"from_file": classmethod(lambda cls, path: _FakeTokenizer())},
    )
    fake_module = ModuleType("tokenizers")
    fake_module.Tokenizer = fake_tokenizer_class
    monkeypatch.setitem(sys.modules, "tokenizers", fake_module)

    counter = E5TokenCounter.load(str(tokenizer_file))

    assert counter.name == str(tokenizer_file.resolve())
    assert counter.count_content("a b c") == 3


def test_load_raises_runtime_error_when_tokenizers_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)  # forces ImportError on import

    with pytest.raises(RuntimeError, match="tokenizers dependency is missing"):
        E5TokenCounter.load("intfloat/multilingual-e5-base")


def test_load_wraps_pretrained_download_failures(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("no internet")

    fake_tokenizer_class = type(
        "FakeTokenizerClass",
        (),
        {"from_pretrained": classmethod(lambda cls, identifier, revision: _raise())},
    )
    fake_module = ModuleType("tokenizers")
    fake_module.Tokenizer = fake_tokenizer_class
    monkeypatch.setitem(sys.modules, "tokenizers", fake_module)

    with pytest.raises(RuntimeError, match="Unable to load the pinned E5 tokenizer"):
        E5TokenCounter.load("intfloat/multilingual-e5-base")


def test_passage_prefix_constant_matches_documented_e5_convention():
    assert E5_PASSAGE_PREFIX == "passage: "
