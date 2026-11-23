from __future__ import annotations


class ImageTooBlurryError(Exception):
    """Raised when Laplacian variance is below threshold."""


class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported by Processing pipeline."""
