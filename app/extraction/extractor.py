from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

import fitz
import pandas as pd
from docx import Document

from app.extraction.constants import (
    PDF_MIN_CHARS_PER_PAGE,
    TEMP_DIR,
    TEXT_FILE_ENCODINGS,
)
from app.extraction.file_types import resolve_content_type
from app.extraction.ocr import ImageAnalyzer

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts plain text from supported file types."""

    def __init__(self, analyzer: ImageAnalyzer, temp_dir: str = TEMP_DIR) -> None:
        self.analyzer = analyzer
        self.temp_dir = temp_dir

    def extract(self, file_path: str, file_type: str) -> str:
        content_type = resolve_content_type(file_path, file_type)

        handlers: dict[str, Callable[[str], str]] = {
            "txt": self.read_text_file,
            "docx": self.extract_docx,
            "doc": self.extract_doc,
            "pdf": self.extract_pdf,
            "image": self.extract_image,
            "xlsx": self.extract_spreadsheet,
            "xls": self.extract_spreadsheet,
        }
        return handlers[content_type](file_path)

    def read_text_file(self, file_path: str) -> str:
        for enc in TEXT_FILE_ENCODINGS:
            try:
                with open(file_path, encoding=enc) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        with open(file_path, "rb") as f:
            return f.read().decode("latin-1", errors="ignore").strip()

    def extract_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join(
            p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()
        ).strip()

    def extract_image(self, file_path: str) -> str:
        return self.analyzer.ocr_file(file_path)

    def extract_pdf(self, file_path: str) -> str:
        native_text = self._extract_pdf_native(file_path)
        if native_text:
            return native_text
        logger.info("No native text found in PDF, falling back to OCR: %s", file_path)
        pages = self.analyzer.pdf_to_bgr_pages(file_path)
        if not pages:
            return ""
        return "\n".join(self.analyzer.ocr_array(p) for p in pages).strip()

    def _extract_pdf_native(self, file_path: str) -> str:
        with fitz.open(file_path) as doc:
            page_texts = [doc.load_page(i).get_text("text") for i in range(len(doc))]
        total_chars = sum(len(t.strip()) for t in page_texts)
        if total_chars < len(page_texts) * PDF_MIN_CHARS_PER_PAGE:
            return ""
        return "\n".join(t.strip() for t in page_texts if t.strip())

    def extract_doc(self, file_path: str) -> str:
        soffice = shutil.which("soffice")
        if not soffice:
            raise RuntimeError("LibreOffice (soffice) is not installed.")

        out_dir = tempfile.mkdtemp(dir=self.temp_dir)
        try:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    out_dir,
                    file_path,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            txt_files = [f for f in os.listdir(out_dir) if f.endswith(".txt")]
            if not txt_files:
                raise RuntimeError(
                    "LibreOffice conversion succeeded but output file missing."
                )
            return self.read_text_file(os.path.join(out_dir, txt_files[0]))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out converting .doc with LibreOffice.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"LibreOffice conversion failed: {(exc.stderr or '').strip()}"
            ) from exc
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def extract_spreadsheet(self, file_path: str) -> str:
        sheets = pd.read_excel(file_path, sheet_name=None)
        lines = []
        for df in sheets.values():
            for _, row in df.iterrows():
                row_text = " ".join(str(v) for v in row if pd.notna(v))
                if row_text.strip():
                    lines.append(row_text)
        return "\n".join(lines).strip()
