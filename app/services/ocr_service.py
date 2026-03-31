from __future__ import annotations
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, List
import boto3
import cv2
import easyocr
import fitz
import numpy as np
import requests
from botocore.exceptions import BotoCoreError, ClientError
from docx import Document

from app.core.config import (
    AWS_S3_ACCESS_KEY,
    AWS_S3_REGION,
    AWS_S3_SECRET_KEY,
    LAPLACIAN_VAR_THRESHOLD,
    OCR_USE_GPU,
    TEMP_DIR,
)

logger = logging.getLogger(__name__)

_S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_MIME = "application/msword"
TXT_MIME = "text/plain"
PDF_MIME = "application/pdf"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
TEXT_FILE_ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")
SUPPORTED_OCR_TYPES = (
    "image/*",
    "application/pdf",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)


class ImageTooBlurryError(Exception):
    """Raised when Laplacian variance is below threshold (MLOps data validation)."""

    pass


class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported by OCR pipeline."""

    pass


class OcrService:
    def __init__(
        self,
        languages: list[str] | None = None,
        blur_threshold: float = LAPLACIAN_VAR_THRESHOLD,
        temp_dir: str = TEMP_DIR,
        use_gpu: bool = OCR_USE_GPU,
    ) -> None:
        self._languages = languages or ["vi", "en"]
        self._blur_threshold = blur_threshold
        self._temp_dir = temp_dir
        self._use_gpu = use_gpu
        self._reader: easyocr.Reader | None = None

    def _get_reader(self) -> easyocr.Reader:
        if self._reader is None:
            logger.info("Initializing EasyOCR reader with gpu=%s", self._use_gpu)
            self._reader = easyocr.Reader(self._languages, gpu=self._use_gpu)
        return self._reader

    def _s3_client(self):
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_S3_ACCESS_KEY or None,
            aws_secret_access_key=AWS_S3_SECRET_KEY or None,
            region_name=AWS_S3_REGION,
        )

    @staticmethod
    def _suffix_for_type(file_type: str) -> str:
        ft = (file_type or "").lower()
        if "image/png" in ft or ft == "image/png":
            return "png"
        if "image/jpeg" in ft or ft == "image/jpeg" or "image/jpg" in ft:
            return "jpg"
        if PDF_MIME in ft or ft == PDF_MIME:
            return "pdf"
        if DOCX_MIME in ft or ft == DOCX_MIME:
            return "docx"
        if DOC_MIME in ft or ft == DOC_MIME:
            return "doc"
        if TXT_MIME in ft or ft == TXT_MIME:
            return "txt"
        return "bin"

    def _download_s3_to_path(self, file_url: str, path: str) -> None:
        m = _S3_URI.match(file_url)
        if not m:
            raise ValueError(f"Invalid S3 URI: {file_url!r}")
        bucket, key = m.group(1), m.group(2)
        try:
            self._s3_client().download_file(bucket, key, path)
        except (ClientError, BotoCoreError) as e:
            logger.exception("S3 download failed bucket=%s key=%s: %s", bucket, key, e)
            raise

    @staticmethod
    def _download_http_to_path(file_url: str, path: str) -> None:
        resp = requests.get(file_url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            if path and os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass

    def download_to_temp(self, file_url: str, file_type: str) -> str:
        ext = self._suffix_for_type(file_type)
        fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=self._temp_dir)
        os.close(fd)

        try:
            if file_url.startswith("s3://"):
                self._download_s3_to_path(file_url, path)
            else:
                self._download_http_to_path(file_url, path)
        except Exception:
            self._safe_unlink(path)
            raise

        return path

    def validate_image_blur(self, image_path: str) -> tuple[bool, float]:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (var >= self._blur_threshold, float(var))

    def _ocr_image(self, image_path: str) -> str:
        reader = self._get_reader()
        results = reader.readtext(image_path)
        return self._join_ocr_text(results)

    @staticmethod
    def _pdf_to_images(pdf_path: str) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        mat = fitz.Matrix(150 / 72, 150 / 72)

        with fitz.open(pdf_path) as doc:
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                rgb = fitz.Pixmap(fitz.csRGB, pix)
                h, w = rgb.height, rgb.width
                img = np.frombuffer(rgb.samples, dtype=np.uint8).reshape(h, w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)

        return images

    def _ocr_image_array(self, img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        reader = self._get_reader()
        results = reader.readtext(img_rgb)
        return self._join_ocr_text(results)

    @staticmethod
    def _join_ocr_text(results: List[Any]) -> str:
        return " ".join(str(r[1]) for r in results if len(r) > 1).strip()

    def _ensure_image_not_blurry(self, image_path: str, context: str = "Image") -> None:
        ok, var = self.validate_image_blur(image_path)
        if not ok:
            raise ImageTooBlurryError(
                f"{context} too blurry (Laplacian variance={var:.1f} < {self._blur_threshold})"
            )

    def _run_pdf_ocr(self, file_path: str) -> str:
        images = self._pdf_to_images(file_path)
        if not images:
            return ""

        fd, first_path = tempfile.mkstemp(suffix=".png", dir=self._temp_dir)
        try:
            os.close(fd)
            cv2.imwrite(first_path, images[0])
            self._ensure_image_not_blurry(first_path, context="First page image/PDF")
        finally:
            self._safe_unlink(first_path)

        texts = [self._ocr_image_array(img) for img in images]
        return "\n".join(texts).strip()

    def _run_image_ocr(self, file_path: str) -> str:
        self._ensure_image_not_blurry(file_path)
        return self._ocr_image(file_path)

    @staticmethod
    def _resolve_content_type(file_path: str, file_type: str) -> str:
        ft = (file_type or "").lower()
        lower_path = file_path.lower()
        if TXT_MIME in ft or lower_path.endswith(".txt"):
            return "txt"
        if DOCX_MIME in ft or lower_path.endswith(".docx"):
            return "docx"
        if DOC_MIME in ft or lower_path.endswith(".doc"):
            return "doc"
        if "pdf" in ft or lower_path.endswith(".pdf"):
            return "pdf"
        if ft.startswith("image/") or lower_path.endswith(IMAGE_EXTENSIONS):
            return "image"
        return "unsupported"

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        for enc in TEXT_FILE_ENCODINGS:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        with open(file_path, "rb") as f:
            return f.read().decode("latin-1", errors="ignore").strip()

    @staticmethod
    def _extract_docx_text(file_path: str) -> str:
        doc = Document(file_path)
        paragraphs = [
            p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()
        ]
        return "\n".join(paragraphs).strip()

    def _extract_doc_text_via_soffice(self, file_path: str) -> str:
        soffice_path = shutil.which("soffice")
        if not soffice_path:
            raise RuntimeError(
                "Cannot process .doc because LibreOffice (soffice) is not installed."
            )

        out_dir = tempfile.mkdtemp(dir=self._temp_dir)
        txt_path = os.path.join(
            out_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt"
        )
        try:
            subprocess.run(
                [
                    soffice_path,
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
            if not os.path.isfile(txt_path):
                raise RuntimeError(
                    "LibreOffice conversion succeeded but output file missing."
                )
            return self._read_text_file(txt_path)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "Timed out while converting .doc with LibreOffice."
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(
                f"Failed to convert .doc with LibreOffice: {stderr}"
            ) from e
        finally:
            try:
                self._safe_unlink(txt_path)
                os.rmdir(out_dir)
            except OSError:
                pass

    def run_ocr(self, file_path: str, file_type: str) -> str:
        content_type = self._resolve_content_type(file_path, file_type)

        if content_type == "unsupported":
            display_type = file_type or "application/octet-stream"
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {display_type}. Supported types: {', '.join(SUPPORTED_OCR_TYPES)}."
            )

        handlers: dict[str, Callable[[str], str]] = {
            "txt": self._read_text_file,
            "docx": self._extract_docx_text,
            "doc": self._extract_doc_text_via_soffice,
            "pdf": self._run_pdf_ocr,
            "image": self._run_image_ocr,
        }
        return handlers[content_type](file_path)

    @staticmethod
    def cleanup_temp(path: str) -> None:
        OcrService._safe_unlink(path)


ocr_service = OcrService()
