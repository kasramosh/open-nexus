from dataclasses import dataclass
from langchain_core.documents import Document
from app.rag.embedder import CHROMA_PERSIST_DIR, EmbeddingConfig, get_vector_store
from app.rag.store import collection_exists

DEFAULT_RETRIEVAL_K = 5


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for the retriever."""

    top_k: int = DEFAULT_RETRIEVAL_K

    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


def retrieve_relevant_chunks(
    query: str,
    collection_name: str = "default",
    embed_config: EmbeddingConfig | None = None,
    persist_directory: str = CHROMA_PERSIST_DIR,
    retriever_config: RetrieverConfig | None = None,
) -> list[Document]:
    """Retrieve relevant chunks from the vector store based on the query."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    
    if not collection_exists(collection_name, persist_directory):
        raise ValueError(f"collection does not exist: {collection_name}")

    vector_store = get_vector_store(
        collection_name=collection_name,
        embed_config=embed_config,
        persist_directory=persist_directory,
    )

    retriever_cfg = retriever_config or RetrieverConfig()
    return vector_store.similarity_search(query, k=retriever_cfg.top_k)
