from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Any

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

@dataclass (frozen=True)
class ChunkingConfig:
    """Configuration for chunking text documents."""
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separators: list[str] = field(
    default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    encoding_name: str = "cl100k_base"
    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")


def build_token_splitter(config: ChunkingConfig | None = None) -> RecursiveCharacterTextSplitter:
    """Builds a RecursiveCharacterTextSplitter based on the provided configuration."""

    cfg = config or ChunkingConfig()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=cfg.encoding_name,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
    )

def chunk_text_to_documents(text: str, config: ChunkingConfig | None = None, source: str = "", 
                            base_metadata: dict[str, Any] | None = None) -> list[Document]:
    """Chunks the input text into a list of Document objects with metadata."""
    if not text or not text.strip():
        return []
    
    cfg = config or ChunkingConfig()
    splitter = build_token_splitter(cfg)
    chunks = splitter.split_text(text)

    metadata_seed = dict(base_metadata) if base_metadata else {}
    documents = []

    for i, chunk in enumerate(chunks):
        metadata = dict(metadata_seed)
        metadata["source"] = source
        metadata["chunk_index"] = i
        documents.append(Document(page_content=chunk, metadata=metadata))

    return documents
