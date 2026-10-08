from app import app
from app.routers import indexing, rag, search, summarize

app.include_router(search.router)
app.include_router(rag.router)
app.include_router(summarize.router)
app.include_router(indexing.router)
