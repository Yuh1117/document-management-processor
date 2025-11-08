from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app import app

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vector = model.encode(req.text).tolist()
    return EmbedResponse(embedding=vector)