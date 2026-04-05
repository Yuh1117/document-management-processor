from __future__ import annotations

import logging
import os

import cv2
import fitz
import numpy as np
from docx import Document as DocxDocument

from app.core.config import (
    LAPLACIAN_VAR_THRESHOLD,
    MIN_CONTRAST_THRESHOLD,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    VALIDATE_ALL_PDF_PAGES,
)
from app.models.validation import ValidationCheck, ValidationReport
from app.services.ocr_service import OcrService

logger = logging.getLogger(__name__)


class DataValidationService:
    @staticmethod
    def _check_image(img: np.ndarray, label: str = "Image") -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        h, w = img.shape[:2]

        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            checks.append(
                ValidationCheck(
                    name="resolution",
                    passed=False,
                    message=f"{label} resolution too low ({w}x{h}), min {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}",
                )
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < LAPLACIAN_VAR_THRESHOLD:
            checks.append(
                ValidationCheck(
                    name="blur",
                    passed=False,
                    message=f"{label} too blurry (variance={blur_var:.1f}, min={LAPLACIAN_VAR_THRESHOLD})",
                )
            )

        contrast = float(np.std(gray))
        if contrast < MIN_CONTRAST_THRESHOLD:
            checks.append(
                ValidationCheck(
                    name="contrast",
                    passed=False,
                    message=f"{label} contrast too low (std={contrast:.1f}, min={MIN_CONTRAST_THRESHOLD})",
                )
            )

        return checks

    def _validate_image(self, path: str) -> list[ValidationCheck]:
        img = cv2.imread(path)
        if img is None:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message="Cannot read image file"
                )
            ]
        return self._check_image(img)

    def _validate_pdf(self, path: str) -> list[ValidationCheck]:
        try:
            doc = fitz.open(path)
        except Exception as e:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message=f"PDF corrupt: {e}"
                )
            ]

        if doc.page_count == 0:
            doc.close()
            return [
                ValidationCheck(
                    name="integrity", passed=False, message="PDF has no pages"
                )
            ]

        checks: list[ValidationCheck] = []
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pages = range(len(doc)) if VALIDATE_ALL_PDF_PAGES else range(min(1, len(doc)))

        for i in pages:
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            rgb = fitz.Pixmap(fitz.csRGB, pix)
            img = np.frombuffer(rgb.samples, dtype=np.uint8).reshape(
                rgb.height, rgb.width, 3
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            checks.extend(self._check_image(img_bgr, label=f"Page {i + 1}"))

        doc.close()
        return checks

    def _validate_docx(self, path: str) -> list[ValidationCheck]:
        try:
            DocxDocument(path)
            return []
        except Exception as e:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message=f"DOCX corrupt: {e}"
                )
            ]

    def _validate_text(self, path: str) -> list[ValidationCheck]:
        if os.path.getsize(path) == 0:
            return [
                ValidationCheck(name="integrity", passed=False, message="File is empty")
            ]
        return []

    def validate(self, file_path: str, file_type: str, doc_id: int) -> ValidationReport:
        content_type = OcrService._resolve_content_type(file_path, file_type)

        handlers = {
            "image": self._validate_image,
            "pdf": self._validate_pdf,
            "docx": self._validate_docx,
            "txt": self._validate_text,
            "doc": self._validate_text,
        }

        if content_type == "unsupported":
            checks = [
                ValidationCheck(
                    name="format", passed=False, message=f"Unsupported: {file_type}"
                )
            ]
        elif content_type in handlers:
            checks = handlers[content_type](file_path)
        else:
            checks = []

        overall = len(checks) == 0 or all(c.passed for c in checks)
        if not overall:
            failed = [c.message or c.name for c in checks if not c.passed]
            logger.warning("Validation FAILED doc_id=%s: %s", doc_id, "; ".join(failed))

        return ValidationReport(
            doc_id=doc_id,
            file_type=file_type or "unknown",
            checks=checks,
            overall_passed=overall,
        )


data_validation_service = DataValidationService()
