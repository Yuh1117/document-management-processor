from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.core.index_bootstrap import IndexBootstrap


@asynccontextmanager
async def lifespan(app: FastAPI):
    IndexBootstrap().ensure_index_exists()
    yield


app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
