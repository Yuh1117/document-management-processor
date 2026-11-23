import re

from app.indexing.constants import CHUNK_OVERLAP, CHUNK_SIZE

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []

        current.extend(words)

        while len(current) >= size:
            chunks.append(" ".join(current[:size]))
            current = current[size - overlap :] if overlap > 0 else current[size:]

    if current:
        chunks.append(" ".join(current))

    return chunks
