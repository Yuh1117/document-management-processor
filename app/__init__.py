from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.index_bootstrap import IndexBootstrap


@asynccontextmanager
async def lifespan(app: FastAPI):
    IndexBootstrap().ensure_index_exists()

    yield

    print("Application is shutting down")


app = FastAPI(lifespan=lifespan)


origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
