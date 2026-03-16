from __future__ import annotations
import os
import tempfile
from typing import List
import cv2
import easyocr
import fitz
import numpy as np
import requests

from app.core.config import (
    LAPLACIAN_VAR_THRESHOLD,
    TEMP_DIR,
)


class ImageTooBlurryError(Exception):
    """Raised when Laplacian variance is below threshold (MLOps data validation)."""

    pass


_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["vi", "en"], gpu=False)
    return _reader


def download_to_temp(file_url: str, file_type: str) -> str:
    resp = requests.get(file_url, timeout=60, stream=True)
    resp.raise_for_status()

    ext = "bin"
    if "image/png" in file_type or file_type == "image/png":
        ext = "png"
    elif (
        "image/jpeg" in file_type
        or file_type == "image/jpeg"
        or "image/jpg" in file_type
    ):
        ext = "jpg"
    elif "application/pdf" in file_type or file_type == "application/pdf":
        ext = "pdf"

    fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=TEMP_DIR)
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return path


def validate_image_blur(image_path: str) -> tuple[bool, float]:
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (var >= LAPLACIAN_VAR_THRESHOLD, float(var))


def _ocr_image(image_path: str) -> str:
    reader = _get_reader()
    results = reader.readtext(image_path)
    return " ".join([r[1] for r in results]).strip()


def _pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.tobytes(), dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images


def _ocr_image_array(img_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reader = _get_reader()
    results = reader.readtext(img_rgb)
    return " ".join([r[1] for r in results]).strip()


def run_ocr(file_path: str, file_type: str) -> str:
    ft = (file_type or "").lower()

    if "pdf" in ft or file_path.lower().endswith(".pdf"):
        images = _pdf_to_images(file_path)
        if not images:
            return ""
        # Validate first page blur
        fd, first_path = tempfile.mkstemp(suffix=".png", dir=TEMP_DIR)
        try:
            os.close(fd)
            cv2.imwrite(first_path, images[0])
            ok, var = validate_image_blur(first_path)
            if not ok:
                raise ImageTooBlurryError(
                    f"First page image/PDF too blurry (Laplacian variance={var:.1f} < {LAPLACIAN_VAR_THRESHOLD})"
                )
        finally:
            try:
                os.unlink(first_path)
            except OSError:
                pass
        texts = [_ocr_image_array(img) for img in images]
        return "\n".join(texts).strip()

    # Image
    ok, var = validate_image_blur(file_path)
    if not ok:
        raise ImageTooBlurryError(
            f"Image too blurry (Laplacian variance={var:.1f} < {LAPLACIAN_VAR_THRESHOLD})"
        )
    return _ocr_image(file_path)


def cleanup_temp(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass
