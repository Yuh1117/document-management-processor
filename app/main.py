from app import app
from app.routers import evaluation, indexing, search, summarize

app.include_router(search.router)
app.include_router(summarize.router)
app.include_router(indexing.router)
app.include_router(evaluation.router)
