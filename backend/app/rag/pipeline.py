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

    def stream_tokens(
        self,
        query: str,
        chunks: list[Document],
        generator_config: generator.GeneratorConfig | None = None,
    ):
        """Stream a cited answer token by token for already-retrieved chunks."""
        return generator.stream_answer(
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

    def list_documents(self) -> list[str]:
        return store.list_collection_sources(
            self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )

    def delete_document(self, source: str) -> None:
        store.delete_document(
            self.config.collection_name,
            source=source,
            persist_directory=self.config.persist_directory,
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_vector_store(self):
        return embedder.get_vector_store(
            collection_name=self.config.collection_name,
            embed_config=self.config.embedding_config,
            persist_directory=self.config.persist_directory,
        )


def _demo() -> None:
    """Minimal end-to-end demo: ingest a file and answer a question.

    Run from the backend/ directory (requires OPENAI_API_KEY in the environment):
        python -m app.rag.pipeline <path-to-pdf-or-txt> "your question"
    """
    import argparse
    import pathlib

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Nexus RAG pipeline demo.")
    parser.add_argument("document", help="Path to a .pdf or .txt file to ingest.")
    parser.add_argument("question", help="Question to ask about the document.")
    parser.add_argument("--collection", default="demo", help="Collection name to use.")
    args = parser.parse_args()

    path = pathlib.Path(args.document)
    pipeline = RAGPipeline(PipelineConfig(collection_name=args.collection))
    if not pipeline.collection_exists():
        pipeline.create_collection()

    if path.suffix.lower() == ".pdf":
        pipeline.ingest_pdf(path.read_bytes(), source=path.name)
    else:
        pipeline.ingest_text(path.read_text(encoding="utf-8"), source=path.name)

    result = pipeline.generate(args.question)
    print("\nAnswer:\n" + result.answer)
    print("\nSources:")
    for source in result.sources:
        page = f", page {source['page_number']}" if source.get("page_number") else ""
        print(f"  - {source['source']}{page}")


if __name__ == "__main__":
    _demo()
