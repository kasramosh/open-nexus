import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.generator import GENERATION_MODEL, GeneratorConfig, generate_answer


def test_generator_config_rejects_temperature_below_zero():
    with pytest.raises(ValueError, match="temperature must be between"):
        GeneratorConfig(temperature=-0.1)


def test_generator_config_rejects_temperature_above_two():
    with pytest.raises(ValueError, match="temperature must be between"):
        GeneratorConfig(temperature=2.1)


def test_generator_config_rejects_non_positive_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GeneratorConfig(max_tokens=0)


def test_generate_answer_rejects_empty_query():
    chunks = [Document(page_content="some content")]
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        generate_answer(query="", chunks=chunks)


def test_generate_answer_rejects_empty_chunks():
    with pytest.raises(ValueError, match="chunks must not be empty"):
        generate_answer(query="What is this?", chunks=[])


@patch("app.rag.generator.ChatOpenAI")
def test_generate_answer_calls_llm_with_config(mock_chat_openai_cls):
    mock_llm = MagicMock(name="ChatOpenAIInstance")
    mock_chat_openai_cls.return_value = mock_llm
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42 [1].")

    chunks = [Document(page_content="The answer is 42.", metadata={"source": "doc.pdf", "page_number": 3, "chunk_index": 0})]

    result = generate_answer(query="What is the answer?", chunks=chunks)

    mock_chat_openai_cls.assert_called_once_with(
        model=GENERATION_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )
    assert result.answer == "The answer is 42 [1]."


@patch("app.rag.generator.ChatOpenAI")
def test_generate_answer_returns_sources_with_metadata(mock_chat_openai_cls):
    mock_llm = MagicMock(name="ChatOpenAIInstance")
    mock_chat_openai_cls.return_value = mock_llm
    mock_llm.invoke.return_value = MagicMock(content="Answer.")

    chunks = [
        Document(page_content="chunk A", metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 0}),
        Document(page_content="chunk B", metadata={"source": "b.pdf", "chunk_index": 2}),
    ]

    result = generate_answer(query="What is this?", chunks=chunks)

    assert len(result.sources) == 2
    assert result.sources[0] == {"source": "a.pdf", "page_number": 1, "chunk_index": 0, "content": "chunk A"}
    assert result.sources[1] == {"source": "b.pdf", "page_number": None, "chunk_index": 2, "content": "chunk B"}


@patch("app.rag.generator.ChatOpenAI")
def test_generate_answer_prompt_contains_query_and_chunk_content(mock_chat_openai_cls):
    mock_llm = MagicMock(name="ChatOpenAIInstance")
    mock_chat_openai_cls.return_value = mock_llm
    mock_llm.invoke.return_value = MagicMock(content="Answer.")

    chunks = [Document(page_content="Relevant passage.", metadata={"source": "book.pdf", "page_number": 5, "chunk_index": 1})]

    generate_answer(query="Tell me about this.", chunks=chunks)

    prompt = mock_llm.invoke.call_args[0][0]
    assert "Tell me about this." in prompt
    assert "Relevant passage." in prompt
    assert "book.pdf" in prompt
