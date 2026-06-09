import os
import sys

import pytest
from unittest.mock import MagicMock, patch

# Ensure the backend package root is on sys.path when pytest runs from a different directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

from app.rag.embedder import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSION,
    EmbeddingConfig,
    embed_documents,
)


def test_embedding_config_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension must be a positive integer"):
        EmbeddingConfig(dimension=0)


def test_embed_documents_rejects_empty_documents():
    with pytest.raises(ValueError, match="no documents provided for embedding"):
        embed_documents([])


@patch("app.rag.embedder.OpenAIEmbeddings")
@patch("app.rag.embedder.Chroma")
def test_embed_documents_builds_and_returns_vector_store(mock_chroma_cls, mock_openai_embeddings_cls):
    mock_embedder = MagicMock(name="OpenAIEmbeddingsInstance")
    mock_openai_embeddings_cls.return_value = mock_embedder

    mock_vector_store = MagicMock(name="ChromaInstance")
    mock_chroma_cls.return_value = mock_vector_store

    documents = [Document(page_content="hello world", metadata={"source": "sample.txt"}), Document(page_content="goodbye cruel world", metadata={"source": "sample.txt"})]
    result = embed_documents(documents, collection_name="test_collection")

    mock_openai_embeddings_cls.assert_called_once_with(
        model="text-embedding-3-small",
        dimension=EMBEDDING_DIMENSION,
    )
    mock_chroma_cls.assert_called_once_with(
        collection_name="test_collection",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=mock_embedder,
    )
    mock_vector_store.add_documents.assert_called_once_with(documents)
    assert result is mock_vector_store
