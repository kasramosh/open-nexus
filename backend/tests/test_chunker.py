import pytest
from unittest.mock import MagicMock, patch

from app.rag.chunker import ChunkingConfig, chunk_text_to_documents, chunk_pdf_to_documents


LONG_TEXT = "Nexus helps users upload documents and ask grounded questions. " * 50


def _make_mock_reader(pages_text: list[str]) -> MagicMock:
    """Return a mock PdfReader whose pages yield the given strings from extract_text()."""
    reader = MagicMock()
    reader.pages = [MagicMock(**{"extract_text.return_value": t}) for t in pages_text]
    return reader

# chunk_text_to_documents
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


# chunk_pdf_to_documents

def test_chunk_pdf_to_documents_returns_empty_for_empty_bytes():
    assert chunk_pdf_to_documents(b"") == []


@patch("app.rag.chunker.pypdf.PdfReader")
def test_chunk_pdf_to_documents_returns_documents(mock_cls):
    mock_cls.return_value = _make_mock_reader([LONG_TEXT])
    docs = chunk_pdf_to_documents(b"fake", source="doc.pdf")

    assert len(docs) > 0
    assert all(doc.page_content for doc in docs)


@patch("app.rag.chunker.pypdf.PdfReader")
def test_chunk_pdf_to_documents_preserves_metadata(mock_cls):
    mock_cls.return_value = _make_mock_reader([LONG_TEXT])
    docs = chunk_pdf_to_documents(b"fake", source="report.pdf", base_metadata={"collection": "demo"})

    assert docs[0].metadata["source"] == "report.pdf"
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["collection"] == "demo"


@patch("app.rag.chunker.pypdf.PdfReader")
def test_chunk_pdf_to_documents_page_numbers_are_1_indexed(mock_cls):
    mock_cls.return_value = _make_mock_reader([LONG_TEXT, LONG_TEXT])
    docs = chunk_pdf_to_documents(b"fake")

    page_numbers = {doc.metadata["page_number"] for doc in docs}
    assert page_numbers == {1, 2}


@patch("app.rag.chunker.pypdf.PdfReader")
def test_chunk_pdf_to_documents_chunk_index_is_global_across_pages(mock_cls):
    mock_cls.return_value = _make_mock_reader([LONG_TEXT, LONG_TEXT])
    docs = chunk_pdf_to_documents(b"fake")

    assert [doc.metadata["chunk_index"] for doc in docs] == list(range(len(docs)))


@patch("app.rag.chunker.pypdf.PdfReader")
def test_chunk_pdf_to_documents_skips_empty_pages(mock_cls):
    mock_cls.return_value = _make_mock_reader([LONG_TEXT, "", LONG_TEXT])
    docs = chunk_pdf_to_documents(b"fake")

    page_numbers = {doc.metadata["page_number"] for doc in docs}
    assert 2 not in page_numbers
    assert page_numbers == {1, 3}

