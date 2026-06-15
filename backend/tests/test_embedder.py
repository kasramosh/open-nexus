import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.embedder import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSION,
    EmbeddingConfig,
    build_embedder,
    embed_documents,
    get_vector_store,
)


def test_embedding_config_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension must be a positive integer"):
        EmbeddingConfig(dimension=0)


@patch("app.rag.embedder.OpenAIEmbeddings")
def test_build_embedder_uses_configured_model_and_dimension(mock_openai_embeddings_cls):
    mock_embedder = MagicMock(name="OpenAIEmbeddingsInstance")
    mock_openai_embeddings_cls.return_value = mock_embedder

    result = build_embedder()

    mock_openai_embeddings_cls.assert_called_once_with(
        model="text-embedding-3-small",
        dimensions=EMBEDDING_DIMENSION,
    )
    assert result is mock_embedder


@patch("app.rag.embedder.build_embedder")
@patch("app.rag.embedder.Chroma")
def test_get_vector_store_builds_chroma_collection(mock_chroma_cls, mock_build_embedder):
    mock_embedder = MagicMock(name="OpenAIEmbeddingsInstance")
    mock_build_embedder.return_value = mock_embedder
    mock_vector_store = MagicMock(name="ChromaInstance")
    mock_chroma_cls.return_value = mock_vector_store

    result = get_vector_store(collection_name="test_collection")

    mock_chroma_cls.assert_called_once_with(
        collection_name="test_collection",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=mock_embedder,
    )
    assert result is mock_vector_store


def test_embed_documents_rejects_empty_documents():
    with pytest.raises(ValueError, match="no documents provided for embedding"):
        embed_documents([])


@patch("app.rag.embedder.get_vector_store")
def test_embed_documents_adds_documents_with_stable_ids(mock_get_vector_store):
    mock_vector_store = MagicMock(name="ChromaInstance")
    mock_get_vector_store.return_value = mock_vector_store

    documents = [
        Document(page_content="hello world", metadata={"source": "sample.txt", "chunk_index": 0}),
        Document(
            page_content="goodbye cruel world",
            metadata={"document_id": "doc-1", "page_number": 2, "chunk_index": 1},
        ),
    ]
    result = embed_documents(documents, collection_name="test_collection")

    mock_get_vector_store.assert_called_once_with(
        collection_name="test_collection",
        persist_directory=CHROMA_PERSIST_DIR,
        embed_config=None,
    )
    mock_vector_store.add_documents.assert_called_once_with(
        documents,
        ids=["sample.txt:chunk:0", "doc-1:page:2:chunk:1"],
    )
    assert result is mock_vector_store
