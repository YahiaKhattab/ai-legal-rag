from dataclasses import dataclass

from legal_rag.ingestion.tokenization import E5TokenCounter


@dataclass
class FakeEncoding:
    ids: list[int]


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool = True) -> FakeEncoding:
        self.calls.append((text, add_special_tokens))
        count = len(text.split()) + (2 if add_special_tokens else 0)
        return FakeEncoding(list(range(count)))


def test_counts_the_exact_prefixed_e5_passage() -> None:
    tokenizer = RecordingTokenizer()
    counter = E5TokenCounter(tokenizer, name="test-e5")

    passage_count = counter.count_passage("نص قانوني")
    content_count = counter.count_content("نص قانوني")

    assert passage_count == 5
    assert content_count == 2
    assert tokenizer.calls == [
        ("passage: نص قانوني", True),
        ("نص قانوني", False),
    ]
    assert counter.name == "test-e5"
