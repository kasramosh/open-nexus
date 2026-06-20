import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.models import HealthResponse
from app.routes import collections, documents, query

app = FastAPI(title="Nexus API")

# Comma-separated list of allowed origins; defaults to the local Vite dev server.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")
