from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class CreateCollectionRequest(APIModel):
    name: str = Field(..., min_length=1, max_length=80)


class CollectionOut(APIModel):
    name: str


class CollectionListOut(APIModel):
    collections: list[CollectionOut]


class DocumentOut(APIModel):
    source: str

class DocumentListOut(APIModel):
    documents: list[DocumentOut]


class SourceChunk(APIModel):
    source: str
    page_number: int | None = Field(None, ge=1)
    chunk_index: int | None = Field(None, ge=0)
    content: str


class QueryRequest(APIModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(APIModel):
    answer: str
    sources: list[SourceChunk]


class MessageResponse(APIModel):
    message: str


class HealthResponse(APIModel):
    status: Literal["ok"]
