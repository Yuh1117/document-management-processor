from dataclasses import dataclass, field
from typing import Any


@dataclass
class Passage:
    document_id: str
    name: str | None
    text: str
    score: float
    copies: list[tuple[str, str | None]] = field(default_factory=list)


def split_windows(text: str, words_per_window: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_window])
        for i in range(0, len(words), words_per_window)
    ]


def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def add_copy(hit: dict[str, Any], document_id: Any, name: str | None) -> None:
    source = hit.get("_source") or {}
    document_id = str(document_id)
    copies = hit.setdefault("_copies", [])
    if document_id != str(source.get("document_id")) and all(
        c[0] != document_id for c in copies
    ):
        copies.append((document_id, name))


def add_copy_to_passage(passage: "Passage", document_id: str, name: str | None) -> None:
    if document_id != passage.document_id and all(
        c[0] != document_id for c in passage.copies
    ):
        passage.copies.append((document_id, name))


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."
