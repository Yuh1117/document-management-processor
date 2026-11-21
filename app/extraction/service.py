from __future__ import annotations

from app.core.config import (
    OCR_USE_GPU,
)
from app.extraction.constants import (
    LAPLACIAN_VAR_THRESHOLD,
    TEMP_DIR,
)
from app.extraction.downloader import FileDownloader
from app.extraction.extractor import TextExtractor
from app.extraction.file_types import resolve_content_type, safe_unlink
from app.extraction.ocr import ImageAnalyzer
from app.extraction.schemas import ValidationReport
from app.extraction.validator import FileValidator


class ExtractionService:
    def __init__(
        self,
        languages: list[str] | None = None,
        blur_threshold: float = LAPLACIAN_VAR_THRESHOLD,
        temp_dir: str = TEMP_DIR,
        use_gpu: bool = OCR_USE_GPU,
    ) -> None:
        analyzer = ImageAnalyzer(languages, blur_threshold, use_gpu)
        self.downloader = FileDownloader(temp_dir)
        self.extractor = TextExtractor(analyzer, temp_dir)
        self.validator = FileValidator(analyzer)

    def is_supported(self, file_type: str) -> bool:
        return resolve_content_type("", file_type) != "unsupported"

    def download_to_temp(self, file_url: str, file_type: str) -> str:
        return self.downloader.download(file_url, file_type)

    def validate(self, file_path: str, file_type: str, doc_id: str) -> ValidationReport:
        return self.validator.validate(file_path, file_type, doc_id)

    def extract_text(self, file_path: str, file_type: str) -> str:
        return self.extractor.extract(file_path, file_type)

    @staticmethod
    def cleanup_temp(path: str) -> None:
        safe_unlink(path)


extraction_service = ExtractionService()
