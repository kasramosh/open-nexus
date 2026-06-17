from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generator import GenerationResult

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_pipeline(exists: bool = True) -> MagicMock:
    """Return a pipeline mock with collection_exists pre-set."""
    m = MagicMock()
    m.collection_exists.return_value = exists
    return m


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Collections ───────────────────────────────────────────────────────────────

@patch("app.routes.collections.RAGPipeline")
def test_create_collection_returns_201(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=False)
    response = client.post("/collections", json={"name": "my-docs"})
    assert response.status_code == 201
    assert response.json() == {"name": "my-docs"}


@patch("app.routes.collections.RAGPipeline")
def test_create_collection_conflict_returns_409(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=True)
    response = client.post("/collections", json={"name": "my-docs"})
    assert response.status_code == 409


@patch("app.routes.collections.RAGPipeline")
def test_list_collections_returns_names(mock_cls):
    pipeline = _mock_pipeline()
    pipeline.get_all_collections.return_value = ["alpha", "beta"]
    mock_cls.return_value = pipeline

    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json() == {"collections": [{"name": "alpha"}, {"name": "beta"}]}


@patch("app.routes.collections.RAGPipeline")
def test_delete_collection_returns_204(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=True)
    response = client.delete("/collections/my-docs")
    assert response.status_code == 204


@patch("app.routes.collections.RAGPipeline")
def test_delete_nonexistent_collection_returns_404(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=False)
    response = client.delete("/collections/my-docs")
    assert response.status_code == 404


# ── Documents ─────────────────────────────────────────────────────────────────

@patch("app.routes.documents.RAGPipeline")
def test_upload_text_document_returns_201(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=True)
    response = client.post(
        "/collections/my-docs/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 201
    assert response.json() == {"source": "notes.txt"}


@patch("app.routes.documents.RAGPipeline")
def test_upload_to_missing_collection_returns_404(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=False)
    response = client.post(
        "/collections/my-docs/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 404


@patch("app.routes.documents.RAGPipeline")
def test_list_documents_returns_sources(mock_cls):
    pipeline = _mock_pipeline(exists=True)
    pipeline.list_documents.return_value = ["a.txt", "b.pdf"]
    mock_cls.return_value = pipeline

    response = client.get("/collections/my-docs/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [{"source": "a.txt"}, {"source": "b.pdf"}]}


@patch("app.routes.documents.RAGPipeline")
def test_list_documents_missing_collection_returns_404(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=False)
    response = client.get("/collections/my-docs/documents")
    assert response.status_code == 404


@patch("app.routes.documents.RAGPipeline")
def test_delete_document_returns_204(mock_cls):
    pipeline = _mock_pipeline(exists=True)
    pipeline.list_documents.return_value = ["notes.txt"]
    mock_cls.return_value = pipeline

    response = client.delete("/collections/my-docs/documents/notes.txt")
    assert response.status_code == 204


@patch("app.routes.documents.RAGPipeline")
def test_delete_missing_document_returns_404(mock_cls):
    pipeline = _mock_pipeline(exists=True)
    pipeline.list_documents.return_value = []
    mock_cls.return_value = pipeline

    response = client.delete("/collections/my-docs/documents/ghost.txt")
    assert response.status_code == 404


# ── Query ─────────────────────────────────────────────────────────────────────

@patch("app.routes.query.RAGPipeline")
def test_query_returns_answer_and_sources(mock_cls):
    pipeline = _mock_pipeline(exists=True)
    pipeline.generate.return_value = GenerationResult(
        answer="Nexus is a RAG platform.",
        sources=[{"source": "a.txt", "page_number": None, "chunk_index": 0, "content": "..."}],
    )
    mock_cls.return_value = pipeline

    response = client.post(
        "/collections/my-docs/query",
        json={"query": "what is nexus?", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Nexus is a RAG platform."
    assert len(data["sources"]) == 1


@patch("app.routes.query.RAGPipeline")
def test_query_missing_collection_returns_404(mock_cls):
    mock_cls.return_value = _mock_pipeline(exists=False)
    response = client.post(
        "/collections/my-docs/query",
        json={"query": "what is nexus?", "top_k": 5},
    )
    assert response.status_code == 404


@patch("app.routes.query.RAGPipeline")
def test_query_empty_collection_returns_422(mock_cls):
    pipeline = _mock_pipeline(exists=True)
    pipeline.generate.side_effect = ValueError("chunks must not be empty.")
    mock_cls.return_value = pipeline

    response = client.post(
        "/collections/my-docs/query",
        json={"query": "what is nexus?", "top_k": 5},
    )
    assert response.status_code == 422
