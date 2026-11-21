"""Pick a representative subset of chunks so a long document fits in one LLM call."""

import numpy as np

from app.summarize.constants import SUMMARY_MMR_LAMBDA


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def select_representative(
    chunks: list[str], vectors: np.ndarray, max_chars: int, separator: str
) -> list[int]:
    """Return indexes of chunks (in document order) that fit into max_chars.

    The first and last chunks are always kept (introduction / conclusion).
    The rest is chosen with MMR: close to the document centroid (central
    content) but not redundant with the chunks already selected.
    """
    vectors = normalize(np.asarray(vectors, dtype=np.float32))
    centroid = normalize(vectors.mean(axis=0, keepdims=True))[0]
    relevance = vectors @ centroid

    selected: list[int] = []
    used = 0

    def try_add(i: int) -> bool:
        nonlocal used
        cost = len(chunks[i]) + (len(separator) if selected else 0)
        if used + cost > max_chars:
            return False
        selected.append(i)
        used += cost
        return True

    for i in dict.fromkeys([0, len(chunks) - 1]):
        try_add(i)

    remaining = [i for i in range(len(chunks)) if i not in selected]
    while remaining:
        if selected:
            redundancy = (vectors[remaining] @ vectors[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        scores = (
            SUMMARY_MMR_LAMBDA * relevance[remaining]
            - (1 - SUMMARY_MMR_LAMBDA) * redundancy
        )
        best = remaining[int(np.argmax(scores))]
        remaining.remove(best)
        try_add(best)  # skipped if too long; smaller chunks may still fit

    return sorted(selected)
