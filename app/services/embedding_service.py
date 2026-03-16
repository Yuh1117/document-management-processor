from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import SENTENCE_TRANSFORMER_MODEL_NAME


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
        return self._model

    def encode_text(self, text: str) -> List[float]:
        model = self.get_model()
        return model.encode(text).tolist()


embedding_service = EmbeddingService()
