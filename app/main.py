import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import indexing, rag, search, summarize
from app.indexing.bootstrap import IndexBootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    IndexBootstrap().ensure_index_exists()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(search.router)
app.include_router(rag.router)
app.include_router(summarize.router)
app.include_router(indexing.router)
