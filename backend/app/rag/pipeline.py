from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.rag import chunker, embedder, generator, retriever, store


@dataclass
class PipelineConfig:
    """Configuration for the entire RAG pipeline."""

    collection_name: str
    persist_directory: str = embedder.CHROMA_PERSIST_DIR
    chunking_config: chunker.ChunkingConfig = field(default_factory=chunker.ChunkingConfig)
    embedding_config: embedder.EmbeddingConfig = field(default_factory=embedder.EmbeddingConfig)
    retriever_config: retriever.RetrieverConfig = field(default_factory=retriever.RetrieverConfig)
    generator_config: generator.GeneratorConfig = field(default_factory=generator.GeneratorConfig)

    def __post_init__(self):
        if not self.collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")


class RAGPipeline:
    """Orchestrates chunking, embedding, retrieval, and generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str = "", base_metadata: dict | None = None):
        documents = chunker.chunk_text_to_documents(
            text,
            config=self.config.chunking_config,
            source=source,
            base_metadata=base_metadata,
        )
        if not documents:
            raise ValueError("No text chunks were created from the input.")
        return embedder.embed_documents(
            documents,
            embed_config=self.config.embedding_config,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def ingest_pdf(self, pdf_bytes: bytes, source: str = "", base_metadata: dict | None = None):
        documents = chunker.chunk_pdf_to_documents(
            pdf_bytes,
            config=self.config.chunking_config,
            source=source,
            base_metadata=base_metadata,
        )
        if not documents:
            raise ValueError("No PDF chunks were created from the input.")
        return embedder.embed_documents(
            documents,
            embed_config=self.config.embedding_config,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def build_documents_from_text(
        self,
        text: str,
        source: str = "",
        base_metadata: dict | None = None,
    ) -> list[Document]:
        return chunker.chunk_text_to_documents(
            text,
            config=self.config.chunking_config,
            source=source,
            base_metadata=base_metadata,
        )

    def build_documents_from_pdf(
        self,
        pdf_bytes: bytes,
        source: str = "",
        base_metadata: dict | None = None,
    ) -> list[Document]:
        return chunker.chunk_pdf_to_documents(
            pdf_bytes,
            config=self.config.chunking_config,
            source=source,
            base_metadata=base_metadata,
        )

    def ingest_documents(self, documents: list[Document]):
        if not documents:
            raise ValueError("No documents provided for ingestion.")
        return embedder.embed_documents(
            documents,
            embed_config=self.config.embedding_config,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        retriever_config: retriever.RetrieverConfig | None = None,
    ) -> list[Document]:
        return retriever.retrieve_relevant_chunks(
            query=query,
            collection_name=self.config.collection_name,
            embed_config=self.config.embedding_config,
            persist_directory=self.config.persist_directory,
            retriever_config=retriever_config or self.config.retriever_config,
        )

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retriever_config: retriever.RetrieverConfig | None = None,
        generator_config: generator.GeneratorConfig | None = None,
    ) -> generator.GenerationResult:
        """Retrieve relevant chunks then generate a cited answer."""
        chunks = self.retrieve(query, retriever_config=retriever_config)
        return generator.generate_answer(
            query=query,
            chunks=chunks,
            config=generator_config or self.config.generator_config,
        )

    # ── Collection management ────────────────────────────────────────────────

    def collection_exists(self) -> bool:
        return store.collection_exists(
            self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def create_collection(self) -> None:
        store.create_collection(
            self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def delete_collection(self) -> None:
        store.delete_collection(
            self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def get_all_collections(self) -> list[str]:
        return store.get_all_collections(
            persist_directory=self.config.persist_directory,
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_vector_store(self):
        return embedder.get_vector_store(
            collection_name=self.config.collection_name,
            embed_config=self.config.embedding_config,
            persist_directory=self.config.persist_directory,
        )
