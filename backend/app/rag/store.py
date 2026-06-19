import chromadb
from app.rag.embedder import CHROMA_PERSIST_DIR


def collection_exists(
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> bool:
    client = chromadb.PersistentClient(path=persist_directory)
    return any(c.name == collection_name for c in client.list_collections())

def create_collection(
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    client.create_collection(name=collection_name)


def delete_collection(
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    client.delete_collection(name=collection_name)


def get_all_collections(
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> list[str]:
    client = chromadb.PersistentClient(path=persist_directory)
    return [c.name for c in client.list_collections()]

def list_collection_sources(
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> list[str]:
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)
    result = collection.get(include=["metadatas"])
    sources = {m.get("source", "") for m in result["metadatas"] if m}
    return sorted(sources - {""})


def delete_document(
    collection_name: str,
    source: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)
    collection.delete(where={"source": source})
