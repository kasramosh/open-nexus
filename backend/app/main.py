from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import HealthResponse
from app.routes import collections, documents, query

app = FastAPI(title="Nexus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")
