from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status

from app.models import (
    COLLECTION_NAME_DESCRIPTION,
    COLLECTION_NAME_PATTERN,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.rag.pipeline import PipelineConfig, RAGPipeline
from app.rag.retriever import RetrieverConfig

router = APIRouter(prefix="/collections", tags=["query"])

CollectionId = Annotated[
    str,
    Path(
        min_length=3,
        max_length=80,
        pattern=COLLECTION_NAME_PATTERN,
        description=COLLECTION_NAME_DESCRIPTION,
    ),
]


@router.post("/{collection_id}/query", response_model=QueryResponse)
def query_collection(collection_id: CollectionId, request: QueryRequest):
    pipeline = RAGPipeline(PipelineConfig(collection_name=collection_id))
    if not pipeline.collection_exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    try:
        result = pipeline.generate(
            query=request.query,
            retriever_config=RetrieverConfig(top_k=request.top_k),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return QueryResponse(
        answer=result.answer,
        sources=[SourceChunk(**s) for s in result.sources],
    )
