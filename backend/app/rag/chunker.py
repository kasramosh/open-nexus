import io
from dataclasses import dataclass, field
from typing import Any

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking text documents."""

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    encoding_name: str = "cl100k_base"

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")


def build_token_splitter(config: ChunkingConfig | None = None) -> RecursiveCharacterTextSplitter:
    """Build a token-aware RecursiveCharacterTextSplitter."""
    cfg = config or ChunkingConfig()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=cfg.encoding_name,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )


def _build_metadata(
    metadata_seed: dict[str, Any],
    source: str,
    chunk_index: int,
    page_number: int | None = None,
) -> dict[str, Any]:
    metadata = {
        **metadata_seed,
        "source": source,
        "chunk_index": chunk_index,
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    return metadata


def chunk_text_to_documents(
    text: str,
    config: ChunkingConfig | None = None,
    source: str = "",
    base_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Chunk text into Document objects with source and chunk metadata."""
    if not text or not text.strip():
        return []

    cfg = config or ChunkingConfig()
    splitter = build_token_splitter(cfg)
    chunks = splitter.split_text(text)

    metadata_seed = dict(base_metadata) if base_metadata else {}
    documents = []

    for i, chunk in enumerate(chunks):
        metadata = _build_metadata(metadata_seed, source=source, chunk_index=i)
        documents.append(Document(page_content=chunk, metadata=metadata))

    return documents


def chunk_pdf_to_documents(
    pdf_bytes: bytes,
    config: ChunkingConfig | None = None,
    source: str = "",
    base_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Extract text from PDF bytes and chunk it into Documents with page metadata."""
    if not pdf_bytes:
        return []

    cfg = config or ChunkingConfig()
    splitter = build_token_splitter(cfg)
    metadata_seed = dict(base_metadata) if base_metadata else {}

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    documents = []
    chunk_index = 0

    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in splitter.split_text(page.extract_text() or ""):
            metadata = _build_metadata(
                metadata_seed,
                source=source,
                chunk_index=chunk_index,
                page_number=page_number,
            )
            documents.append(Document(page_content=chunk, metadata=metadata))
            chunk_index += 1

    return documents
