from __future__ import annotations

import logging
from typing import Any

import cv2
import easyocr
import fitz
import numpy as np

from app.core.config import (
    OCR_USE_GPU,
)
from app.extraction.constants import (
    LAPLACIAN_VAR_THRESHOLD,
    MIN_CONTRAST_THRESHOLD,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)
from app.extraction.schemas import ValidationCheck

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Low-level image operations: blur, contrast, OCR via EasyOCR."""

    PDF_RASTER_MATRIX = fitz.Matrix(150 / 72, 150 / 72)

    def __init__(
        self,
        languages: list[str] | None = None,
        blur_threshold: float = LAPLACIAN_VAR_THRESHOLD,
        use_gpu: bool = OCR_USE_GPU,
    ) -> None:
        self.languages = languages or ["vi", "en"]
        self.blur_threshold = blur_threshold
        self.use_gpu = use_gpu
        self.reader = self.load_reader()

    def load_reader(self) -> easyocr.Reader:
        logger.info("Initializing EasyOCR (gpu=%s)", self.use_gpu)
        return easyocr.Reader(self.languages, gpu=self.use_gpu)

    def ocr_file(self, image_path: str) -> str:
        return self.join(self.reader.readtext(image_path))

    def ocr_array(self, img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.join(self.reader.readtext(img_rgb))

    @staticmethod
    def join(results: list[Any]) -> str:
        return " ".join(str(r[1]) for r in results if len(r) > 1).strip()

    def pdf_to_bgr_pages(self, pdf_path: str) -> list[np.ndarray]:
        with fitz.open(pdf_path) as doc:
            return [self.page_to_bgr(doc, i) for i in range(len(doc))]

    @classmethod
    def page_to_bgr(cls, doc: fitz.Document, index: int) -> np.ndarray:
        pix = doc.load_page(index).get_pixmap(matrix=cls.PDF_RASTER_MATRIX, alpha=False)
        rgb = fitz.Pixmap(fitz.csRGB, pix)
        img = np.frombuffer(rgb.samples, dtype=np.uint8).reshape(
            rgb.height, rgb.width, 3
        )
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def image_quality_checks(
        self, img_bgr: np.ndarray, label: str = "Image"
    ) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        h, w = img_bgr.shape[:2]

        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            checks.append(
                ValidationCheck(
                    name="resolution",
                    passed=False,
                    message=f"{label} resolution too low ({w}x{h}), min {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}",
                )
            )

        gray = (
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if len(img_bgr.shape) == 3
            else img_bgr
        )
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))

        if blur_var < self.blur_threshold:
            checks.append(
                ValidationCheck(
                    name="blur",
                    passed=False,
                    message=f"{label} too blurry (variance={blur_var:.1f}, min={self.blur_threshold})",
                )
            )

        if contrast < MIN_CONTRAST_THRESHOLD:
            checks.append(
                ValidationCheck(
                    name="contrast",
                    passed=False,
                    message=f"{label} contrast too low (std={contrast:.1f}, min={MIN_CONTRAST_THRESHOLD})",
                )
            )

        return checks
