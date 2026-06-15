from fastapi import APIRouter, HTTPException, status

from app.models import CreateCollectionRequest, CollectionListOut, CollectionOut
from app.rag.pipeline import PipelineConfig, RAGPipeline

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CollectionOut)
def create_collection(request: CreateCollectionRequest):
    pipeline = RAGPipeline(PipelineConfig(collection_name=request.name))
    if pipeline.collection_exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Collection '{request.name}' already exists.")
    pipeline.create_collection()
    return CollectionOut(name=request.name)


@router.get("", response_model=CollectionListOut)
def list_collections():
    pipeline = RAGPipeline(PipelineConfig(collection_name="_"))
    names = pipeline.get_all_collections()
    return CollectionListOut(collections=[CollectionOut(name=n) for n in names])


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str):
    pipeline = RAGPipeline(PipelineConfig(collection_name=collection_id))
    if not pipeline.collection_exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{collection_id}' not found.")
    pipeline.delete_collection()
