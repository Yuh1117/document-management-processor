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


class OcrService:
    def __init__(
        self,
        languages: list[str] | None = None,
        blur_threshold: float = LAPLACIAN_VAR_THRESHOLD,
        temp_dir: str = TEMP_DIR,
    ) -> None:
        self._languages = languages or ["vi", "en"]
        self._blur_threshold = blur_threshold
        self._temp_dir = temp_dir
        self._reader: easyocr.Reader | None = None

    def _get_reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self._languages, gpu=False)
        return self._reader

    def download_to_temp(self, file_url: str, file_type: str) -> str:
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

        fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=self._temp_dir)
        with os.fdopen(fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
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
        return " ".join([r[1] for r in results]).strip()

    @staticmethod
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

    def _ocr_image_array(self, img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        reader = self._get_reader()
        results = reader.readtext(img_rgb)
        return " ".join([r[1] for r in results]).strip()

    def run_ocr(self, file_path: str, file_type: str) -> str:
        ft = (file_type or "").lower()

        if "pdf" in ft or file_path.lower().endswith(".pdf"):
            images = self._pdf_to_images(file_path)
            if not images:
                return ""
            fd, first_path = tempfile.mkstemp(suffix=".png", dir=self._temp_dir)
            try:
                os.close(fd)
                cv2.imwrite(first_path, images[0])
                ok, var = self.validate_image_blur(first_path)
                if not ok:
                    raise ImageTooBlurryError(
                        f"First page image/PDF too blurry (Laplacian variance={var:.1f} < {self._blur_threshold})"
                    )
            finally:
                try:
                    os.unlink(first_path)
                except OSError:
                    pass
            texts = [self._ocr_image_array(img) for img in images]
            return "\n".join(texts).strip()

        ok, var = self.validate_image_blur(file_path)
        if not ok:
            raise ImageTooBlurryError(
                f"Image too blurry (Laplacian variance={var:.1f} < {self._blur_threshold})"
            )
        return self._ocr_image(file_path)

    @staticmethod
    def cleanup_temp(path: str) -> None:
        try:
            if path and os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


ocr_service = OcrService()
