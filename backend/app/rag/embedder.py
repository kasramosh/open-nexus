from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core import Document

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHROMA_PERSIST_DIR = "./chroma_db"

@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding process."""
    model:str = "text-embedding-3-small"
    dimension: int = EMBEDDING_DIMENSION
    
    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

def embed_documents(documents: list[Document],  
                    embed_config: EmbeddingConfig | None = None, 
                    collection_name: str = "default") -> Chroma:
    """Embeds a list of documents and stores them in a Chroma vector store."""
    
    if not documents:
        raise ValueError("no documents provided for embedding.")
    
    emcfg = embed_config or EmbeddingConfig()
    
    embedder = OpenAIEmbeddings(model=emcfg.model, dimension=emcfg.dimension)
    vector_store = Chroma(collection_name=collection_name, 
                          persist_directory=CHROMA_PERSIST_DIR, 
                          embedding_function=embedder)
    
    vector_store.add_documents(documents)
    
    return vector_store
