from dataclasses import dataclass, field
from xmlrpc import client
import chromadb

from langchain_core.documents import Document
from app.rag import chunker, embedder, retriever 


@dataclass
class PipelineConfig:
    """Configuration for the entire RAG pipeline."""

    collection_name: str
    persist_directory: str = chunker.CHROMA_PERSIST_DIR
    chunking_config: chunker.ChunkingConfig = field(default_factory=chunker.ChunkingConfig)
    embedding_config: embedder.EmbeddingConfig = field(default_factory=embedder.EmbeddingConfig)
    retriever_config: retriever.RetrieverConfig = field(default_factory=retriever.RetrieverConfig)

    def __post_init__(self):
        if not self.collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        

class RAGPipeline:
    """A pipeline that integrates chunking, embedding, and retrieval for RAG."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        
        
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


    def retrieve(
        self,
        query: str,
        retriever_config: retriever.RetrieverConfig | None = None,
    ) -> list[Document]:
        retriever_cfg = retriever_config or self.config.retriever_config

        return retriever.retrieve_relevant_chunks(
            query=query,
            collection_name=self.config.collection_name,
            embed_config=self.config.embedding_config,
            persist_directory=self.config.persist_directory,
            retriever_config=retriever_cfg,
        )
        
    
    def get_vector_store(self):
        return embedder.get_vector_store(
            collection_name=self.config.collection_name,
            embed_config=self.config.embedding_config,
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
        
        
    # TODO: add store.py and add these functions to it
    
    # def collection_exists(self) -> bool:
    #     return retriever.collection_exists(
    #         self.config.collection_name,
    #         persist_directory=self.config.persist_directory,
    #     )
        
    
    # def delete_collection(self) -> None:
    #     client = chromadb.PersistentClient(path=self.config.persist_directory)
    #     client.delete_collection(name=self.config.collection_name)
        
        
    # def get_all_collections(self) -> list[str]:
    #     client = chromadb.PersistentClient(path=self.config.persist_directory)
    #     return [collection.name for collection in client.list_collections()]