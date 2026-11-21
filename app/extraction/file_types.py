from __future__ import annotations

import os

from app.extraction.constants import (
    EXT_TO_CONTENT_TYPE,
    MIME_TO_CONTENT_TYPE,
    MIME_TO_EXT,
)


def safe_unlink(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def suffix_for_type(file_type: str) -> str:
    ft = (file_type or "").lower()
    for mime, ext in MIME_TO_EXT.items():
        if mime in ft:
            return ext
    return "bin"


def resolve_content_type(file_path: str, file_type: str) -> str:
    ft = (file_type or "").lower()
    if ft.startswith("image/"):
        return "image"
    for mime, content_type in MIME_TO_CONTENT_TYPE.items():
        if mime in ft:
            return content_type
    ext = os.path.splitext(file_path.lower())[1]
    return EXT_TO_CONTENT_TYPE.get(ext, "unsupported")
