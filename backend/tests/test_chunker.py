import pytest

from app.rag.chunker import ChunkingConfig, chunk_text_to_documents


def test_chunking_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)

    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_chunk_text_to_documents_returns_empty_list_for_blank_input():
    docs = chunk_text_to_documents("   \n\t  ")
    assert docs == []


def test_chunk_text_to_documents_returns_documents():
    text = "Nexus helps users upload documents and ask grounded questions. " * 50
    docs = chunk_text_to_documents(text, source="sample.txt")

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)


def test_chunk_text_to_documents_preserves_metadata():
    text = "Nexus helps users upload documents and ask grounded questions. " * 50
    docs = chunk_text_to_documents(
        text,
        source="sample.txt",
        base_metadata={"collection": "demo"},
    )

    assert docs[0].metadata["source"] == "sample.txt"
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["collection"] == "demo"

def test_chunk_text_to_documents_with_custom_config():
    text = "Nexus helps users upload documents and ask grounded questions. " * 50
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    docs = chunk_text_to_documents(text, config=config)

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)
    assert [doc.metadata["chunk_index"] for doc in docs] == list(range(len(docs)))

