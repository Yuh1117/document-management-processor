"""Clean extracted text (PDF/DOCX/OCR) before sending it to the LLM."""

import logging
import re
from collections import Counter

from app.summarize.constants import (
    SUMMARY_CLEAN_MIN_KEPT_RATIO,
    SUMMARY_REPEATED_LINE_MAX_CHARS,
    SUMMARY_REPEATED_LINE_MIN_COUNT,
)

logger = logging.getLogger(__name__)

PAGE_NUMBER_LINE = re.compile(
    r"^(?:(?:page|trang)\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?"
    r"(?:\s*(?:/|of|trên)\s*\d{1,4})?$",
    re.IGNORECASE,
)
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
SPACES = re.compile(r"[ \t ]+")
BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def repeated_lines(lines: list[str]) -> set[str]:
    """Short lines repeated verbatim many times: page headers/footers.

    Lines that only differ by a number are kept on purpose, so that
    "Điều 1.", "Chương 2" and similar headings are never removed.
    """
    counts = Counter(
        l for l in lines if l and len(l) <= SUMMARY_REPEATED_LINE_MAX_CHARS
    )
    return {l for l, n in counts.items() if n >= SUMMARY_REPEATED_LINE_MIN_COUNT}


def clean_text(text: str) -> str:
    base = normalize_whitespace(HYPHEN_BREAK.sub(r"\1\2", text))
    lines = base.split("\n")
    drop = repeated_lines(lines)
    kept = [l for l in lines if not PAGE_NUMBER_LINE.match(l) and l not in drop]
    cleaned = BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()

    # Safety net: never let header/footer removal eat the actual content.
    if len(cleaned) < SUMMARY_CLEAN_MIN_KEPT_RATIO * len(base):
        logger.warning(
            "Cleaning removed too much (%d -> %d chars), keeping whitespace-only cleanup",
            len(base),
            len(cleaned),
        )
        return base
    return cleaned
