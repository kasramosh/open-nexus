from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.generator import GenerationResult
from app.rag.pipeline import PipelineConfig, RAGPipeline


def _pipeline() -> RAGPipeline:
    return RAGPipeline(PipelineConfig(collection_name="demo"))


# ── Config ─────────────────────────────────────────────────────────────────────

def test_pipeline_config_rejects_blank_collection_name():
    with pytest.raises(ValueError, match="collection_name must be a non-empty string"):
        PipelineConfig(collection_name="   ")


# ── Ingestion ──────────────────────────────────────────────────────────────────

@patch("app.rag.pipeline.embedder")
@patch("app.rag.pipeline.chunker")
def test_ingest_text_chunks_then_embeds(mock_chunker, mock_embedder):
    docs = [Document(page_content="chunk", metadata={"source": "a.txt", "chunk_index": 0})]
    mock_chunker.chunk_text_to_documents.return_value = docs

    _pipeline().ingest_text("hello world", source="a.txt")

    mock_chunker.chunk_text_to_documents.assert_called_once()
    args, kwargs = mock_embedder.embed_documents.call_args
    assert args[0] == docs
    assert kwargs["collection_name"] == "demo"


@patch("app.rag.pipeline.embedder")
@patch("app.rag.pipeline.chunker")
def test_ingest_text_raises_when_no_chunks(mock_chunker, mock_embedder):
    mock_chunker.chunk_text_to_documents.return_value = []

    with pytest.raises(ValueError, match="No text chunks"):
        _pipeline().ingest_text("")

    mock_embedder.embed_documents.assert_not_called()


@patch("app.rag.pipeline.embedder")
@patch("app.rag.pipeline.chunker")
def test_ingest_pdf_raises_when_no_chunks(mock_chunker, mock_embedder):
    mock_chunker.chunk_pdf_to_documents.return_value = []

    with pytest.raises(ValueError, match="No PDF chunks"):
        _pipeline().ingest_pdf(b"")

    mock_embedder.embed_documents.assert_not_called()


# ── Retrieval & generation ─────────────────────────────────────────────────────

@patch("app.rag.pipeline.generator")
@patch("app.rag.pipeline.retriever")
def test_generate_retrieves_then_generates(mock_retriever, mock_generator):
    chunks = [Document(page_content="ctx", metadata={"source": "a.txt", "chunk_index": 0})]
    mock_retriever.retrieve_relevant_chunks.return_value = chunks
    mock_generator.generate_answer.return_value = GenerationResult(answer="ok", sources=[])

    result = _pipeline().generate("what is nexus?")

    mock_retriever.retrieve_relevant_chunks.assert_called_once()
    _, kwargs = mock_generator.generate_answer.call_args
    assert kwargs["chunks"] == chunks
    assert result.answer == "ok"


@patch("app.rag.pipeline.generator")
def test_stream_tokens_delegates_to_generator(mock_generator):
    chunks = [Document(page_content="ctx")]
    mock_generator.stream_answer.return_value = iter(["a", "b"])

    tokens = list(_pipeline().stream_tokens("q", chunks))

    mock_generator.stream_answer.assert_called_once()
    assert tokens == ["a", "b"]


# ── Collection management ───────────────────────────────────────────────────────

@patch("app.rag.pipeline.store")
def test_collection_management_delegates_to_store(mock_store):
    mock_store.collection_exists.return_value = True
    mock_store.get_all_collections.return_value = ["demo"]
    mock_store.list_collection_sources.return_value = ["a.txt"]

    pipeline = _pipeline()

    assert pipeline.collection_exists() is True
    assert pipeline.get_all_collections() == ["demo"]
    assert pipeline.list_documents() == ["a.txt"]

    pipeline.create_collection()
    pipeline.delete_collection()
    pipeline.delete_document("a.txt")

    mock_store.create_collection.assert_called_once_with("demo", persist_directory=pipeline.config.persist_directory)
    mock_store.delete_collection.assert_called_once_with("demo", persist_directory=pipeline.config.persist_directory)
    mock_store.delete_document.assert_called_once_with(
        "demo", source="a.txt", persist_directory=pipeline.config.persist_directory
    )
