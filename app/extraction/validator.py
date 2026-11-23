from __future__ import annotations

import logging
import os
from collections.abc import Callable

import cv2
import fitz
import pandas as pd
from docx import Document

from app.extraction.constants import (
    PDF_MIN_CHARS_PER_PAGE,
    VALIDATE_ALL_PDF_PAGES,
)
from app.extraction.file_types import resolve_content_type
from app.extraction.ocr import ImageAnalyzer
from app.extraction.schemas import ValidationCheck, ValidationReport

logger = logging.getLogger(__name__)


class FileValidator:
    """Runs quality checks and returns a ValidationReport."""

    def __init__(self, analyzer: ImageAnalyzer) -> None:
        self.analyzer = analyzer

    def validate(self, file_path: str, file_type: str, doc_id: str) -> ValidationReport:
        content_type = resolve_content_type(file_path, file_type)
        checks = self.run_checks(file_path, file_type, content_type)
        overall = not checks or all(c.passed for c in checks)

        if not overall:
            failed = [c.message or c.name for c in checks if not c.passed]
            logger.warning("Validation FAILED doc_id=%s: %s", doc_id, "; ".join(failed))

        return ValidationReport(
            doc_id=doc_id,
            file_type=file_type or "unknown",
            checks=checks,
            overall_passed=overall,
        )

    def run_checks(
        self, file_path: str, file_type: str, content_type: str
    ) -> list[ValidationCheck]:
        if content_type == "unsupported":
            return [
                ValidationCheck(
                    name="format", passed=False, message=f"Unsupported: {file_type}"
                )
            ]

        handlers: dict[str, Callable[[str], list[ValidationCheck]]] = {
            "image": self.check_image,
            "pdf": self.check_pdf,
            "docx": self.check_docx,
            "txt": self.check_text,
            "doc": self.check_text,
            "xlsx": self.check_spreadsheet,
            "xls": self.check_spreadsheet,
        }
        return handlers.get(content_type, lambda _: [])(file_path)

    def check_image(self, path: str) -> list[ValidationCheck]:
        img = cv2.imread(path)
        if img is None:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message="Cannot read image file"
                )
            ]
        return self.analyzer.image_quality_checks(img)

    def check_pdf(self, path: str) -> list[ValidationCheck]:
        try:
            doc = fitz.open(path)
        except Exception as exc:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message=f"PDF corrupt: {exc}"
                )
            ]

        with doc:
            if doc.page_count == 0:
                return [
                    ValidationCheck(
                        name="integrity", passed=False, message="PDF has no pages"
                    )
                ]

            if self._pdf_has_native_text(doc):
                return []

            pages = (
                range(len(doc)) if VALIDATE_ALL_PDF_PAGES else range(min(1, len(doc)))
            )
            checks: list[ValidationCheck] = []
            for i in pages:
                checks.extend(
                    self.analyzer.image_quality_checks(
                        self.analyzer.page_to_bgr(doc, i), label=f"Page {i + 1}"
                    )
                )
            return checks

    def _pdf_has_native_text(self, doc: fitz.Document) -> bool:
        total_chars = sum(
            len(doc.load_page(i).get_text("text").strip()) for i in range(len(doc))
        )
        return total_chars >= len(doc) * PDF_MIN_CHARS_PER_PAGE

    @staticmethod
    def check_docx(path: str) -> list[ValidationCheck]:
        try:
            Document(path)
            return []
        except Exception as exc:
            return [
                ValidationCheck(
                    name="integrity", passed=False, message=f"DOCX corrupt: {exc}"
                )
            ]

    @staticmethod
    def check_text(path: str) -> list[ValidationCheck]:
        if os.path.getsize(path) == 0:
            return [
                ValidationCheck(name="integrity", passed=False, message="File is empty")
            ]
        return []

    @staticmethod
    def check_spreadsheet(path: str) -> list[ValidationCheck]:
        try:
            pd.read_excel(path, nrows=1)
            return []
        except Exception as exc:
            return [
                ValidationCheck(
                    name="integrity",
                    passed=False,
                    message=f"Spreadsheet corrupt: {exc}",
                )
            ]
