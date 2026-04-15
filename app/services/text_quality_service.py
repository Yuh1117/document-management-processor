from __future__ import annotations

import logging
import unicodedata

from app.core.config import TEXT_MAX_INVALID_CHAR_RATIO, TEXT_MIN_QUALITY_SCORE

logger = logging.getLogger(__name__)


class TextQualityService:
    def compute_detailed_metrics(self, text: str) -> dict:
        total = len(text) if text else 0
        words = text.split() if text else []
        clean = self.clean_char_count(text)
        invalid = self.invalid_control_char_count(text)

        return {
            "text_quality_score": int(clean / total * 100) if total else 0,
            "invalid_char_ratio": round(invalid / total, 4) if total else 0.0,
            "avg_word_length": self.avg_word_length(words),
            "total_chars": total,
            "total_words": len(words),
        }

    def check_quality_alert(self, metrics: dict, doc_id: int) -> bool:
        alerts = []
        if metrics.get("text_quality_score", 0) < TEXT_MIN_QUALITY_SCORE:
            alerts.append(
                f"quality_score={metrics['text_quality_score']} < {TEXT_MIN_QUALITY_SCORE}"
            )
        if metrics.get("invalid_char_ratio", 0) > TEXT_MAX_INVALID_CHAR_RATIO:
            alerts.append(
                f"invalid_chars={metrics['invalid_char_ratio']:.4f} > {TEXT_MAX_INVALID_CHAR_RATIO}"
            )

        if alerts:
            logger.warning("Text quality alert doc_id=%s: %s", doc_id, "; ".join(alerts))
        return bool(alerts)

    @staticmethod
    def clean_char_count(text: str) -> int:
        return sum(1 for c in text if c.isalnum() or c.isspace()) if text else 0

    @staticmethod
    def invalid_control_char_count(text: str) -> int:
        if not text:
            return 0
        return sum(
            1
            for c in text
            if unicodedata.category(c).startswith("C") and c not in ("\n", "\r", "\t")
        )

    @staticmethod
    def avg_word_length(words: list[str]) -> float:
        if not words:
            return 0.0
        return round(sum(len(w) for w in words) / len(words), 2)


text_quality_service = TextQualityService()
