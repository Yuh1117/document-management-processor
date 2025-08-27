from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vector = model.encode(req.text).tolist()
    return {"embedding": vector}