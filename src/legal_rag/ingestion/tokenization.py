"""Exact token counting for the selected embedding model."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

DEFAULT_E5_TOKENIZER = "intfloat/multilingual-e5-base"
DEFAULT_E5_REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"
E5_PASSAGE_PREFIX = "passage: "


class TokenCounter(Protocol):
    """Small interface that keeps chunking independent of model libraries."""

    @property
    def name(self) -> str: ...

    def count_passage(self, text: str) -> int: ...

    def count_content(self, text: str) -> int: ...


class E5TokenCounter:
    """Count tokens exactly as multilingual E5 will receive them."""

    def __init__(self, tokenizer: Any, *, name: str) -> None:
        self._tokenizer = tokenizer
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def load(
        cls,
        identifier: str = DEFAULT_E5_TOKENIZER,
        *,
        revision: str = DEFAULT_E5_REVISION,
    ) -> E5TokenCounter:
        """Load a local tokenizer JSON or a pinned public model tokenizer."""

        try:
            tokenizers_module = importlib.import_module("tokenizers")
        except ImportError as error:
            raise RuntimeError(
                "The tokenizers dependency is missing; reinstall the project dependencies"
            ) from error
        tokenizer_class = tokenizers_module.Tokenizer
        local_path = Path(identifier)
        if local_path.is_file():
            tokenizer = tokenizer_class.from_file(str(local_path))
            name = str(local_path.resolve())
        else:
            try:
                tokenizer = tokenizer_class.from_pretrained(identifier, revision=revision)
            except Exception as error:
                raise RuntimeError(
                    "Unable to load the pinned E5 tokenizer. Connect once to download "
                    "its public tokenizer files, or configure a local tokenizer.json path."
                ) from error
            name = f"{identifier}@{revision}"
        return cls(tokenizer, name=name)

    def count_passage(self, text: str) -> int:
        return len(self._tokenizer.encode(f"{E5_PASSAGE_PREFIX}{text}").ids)

    def count_content(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
