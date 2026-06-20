from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

GENERATION_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class GeneratorConfig:
    model: str = GENERATION_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024

    def __post_init__(self):
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]


def _build_prompt(query: str, chunks: list[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number")
        citation = f"[{i}] {source}" + (f", page {page}" if page else "")
        context_parts.append(f"{citation}\n{doc.page_content}")

    context = "\n\n".join(context_parts)
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources by their bracketed number after each claim. "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def build_sources(chunks: list[Document]) -> list[dict]:
    """Project retrieved chunks into the JSON-serializable source shape."""
    return [
        {
            "source": doc.metadata.get("source", ""),
            "page_number": doc.metadata.get("page_number"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "content": doc.page_content,
        }
        for doc in chunks
    ]


def generate_answer(
    query: str,
    chunks: list[Document],
    config: GeneratorConfig | None = None,
) -> GenerationResult:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not chunks:
        raise ValueError("chunks must not be empty.")

    cfg = config or GeneratorConfig()
    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)

    prompt = _build_prompt(query, chunks)
    response = llm.invoke(prompt)

    return GenerationResult(answer=response.content, sources=build_sources(chunks))


def stream_answer(
    query: str,
    chunks: list[Document],
    config: GeneratorConfig | None = None,
) -> Iterator[str]:
    """Stream the generated answer token by token."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not chunks:
        raise ValueError("chunks must not be empty.")

    cfg = config or GeneratorConfig()
    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)

    prompt = _build_prompt(query, chunks)
    for piece in llm.stream(prompt):
        text = piece.content
        if isinstance(text, str) and text:
            yield text
