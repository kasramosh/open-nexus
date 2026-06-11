import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.embedder import CHROMA_PERSIST_DIR
from app.rag.retriever import DEFAULT_RETRIEVAL_K, RetrieverConfig, retrieve_relevant_chunks


def test_retriever_config_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        RetrieverConfig(top_k=0)


def test_retrieve_relevant_chunks_rejects_empty_query():
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retrieve_relevant_chunks(query="")


@patch("app.rag.retriever.collection_exists")
def test_retrieve_relevant_chunks_rejects_missing_collection(mock_collection_exists):
    mock_collection_exists.return_value = False

    with pytest.raises(ValueError, match="collection does not exist: missing"):
        retrieve_relevant_chunks(query="test query", collection_name="missing")


@patch("app.rag.retriever.collection_exists")
@patch("app.rag.retriever.get_vector_store")
def test_retrieve_relevant_chunks_calls_similarity_search(
    mock_get_vector_store,
    mock_collection_exists,
):
    mock_collection_exists.return_value = True
    mock_vector_store = MagicMock(name="ChromaInstance")
    mock_get_vector_store.return_value = mock_vector_store
    mock_vector_store.similarity_search.return_value = [Document(page_content="chunk1"), Document(page_content="chunk2")]

    result = retrieve_relevant_chunks(query="test query", collection_name="test_collection")

    mock_get_vector_store.assert_called_once_with(
        collection_name="test_collection",
        embed_config=None,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    mock_vector_store.similarity_search.assert_called_once_with("test query", k=DEFAULT_RETRIEVAL_K)
    assert result == [Document(page_content="chunk1"), Document(page_content="chunk2")]
