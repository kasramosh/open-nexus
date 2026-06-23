import os
from dataclasses import dataclass

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHROMA_PERSIST_DIR = "./chroma_db"


def get_chroma_client(persist_directory: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    host = os.getenv("CHROMA_HOST")
    if host:
        port = int(os.getenv("CHROMA_PORT", "8000"))
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.PersistentClient(path=persist_directory)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding process."""

    model: str = EMBEDDING_MODEL
    dimension: int = EMBEDDING_DIMENSION

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError("dimension must be a positive integer.")


def build_embedder(embed_config: EmbeddingConfig | None = None) -> OpenAIEmbeddings:
    """Build the OpenAI embedding client used by Chroma."""
    emcfg = embed_config or EmbeddingConfig()
    return OpenAIEmbeddings(model=emcfg.model, dimensions=emcfg.dimension)


def get_vector_store(
    collection_name: str = "default",
    embed_config: EmbeddingConfig | None = None,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Open a Chroma collection with the configured embedding function."""
    return Chroma(
        collection_name=collection_name,
        client=get_chroma_client(persist_directory),
        embedding_function=build_embedder(embed_config),
    )


def _document_id(document: Document, fallback_index: int) -> str:
    metadata = document.metadata
    document_key = metadata.get("document_id") or metadata.get("source") or "document"
    chunk_index = metadata.get("chunk_index", fallback_index)
    page_number = metadata.get("page_number")

    if page_number is None:
        return f"{document_key}:chunk:{chunk_index}"

    return f"{document_key}:page:{page_number}:chunk:{chunk_index}"


def embed_documents(
    documents: list[Document],
    embed_config: EmbeddingConfig | None = None,
    collection_name: str = "default",
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Embed documents and store them in a Chroma vector store."""
    if not documents:
        raise ValueError("no documents provided for embedding.")

    vector_store = get_vector_store(
        collection_name=collection_name,
        embed_config=embed_config,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(
        documents,
        ids=[_document_id(document, i) for i, document in enumerate(documents)],
    )

    return vector_store
