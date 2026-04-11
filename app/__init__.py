import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.index_bootstrap import IndexBootstrap


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
