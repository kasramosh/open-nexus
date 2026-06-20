from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Path, UploadFile, status

from app.models import (
    COLLECTION_NAME_DESCRIPTION,
    COLLECTION_NAME_PATTERN,
    DocumentListOut,
    DocumentOut,
)
from app.rag.pipeline import PipelineConfig, RAGPipeline

router = APIRouter(prefix="/collections", tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

CollectionId = Annotated[
    str,
    Path(
        min_length=3,
        max_length=80,
        pattern=COLLECTION_NAME_PATTERN,
        description=COLLECTION_NAME_DESCRIPTION,
    ),
]


def _require_collection(collection_id: str) -> RAGPipeline:
    pipeline = RAGPipeline(PipelineConfig(collection_name=collection_id))
    if not pipeline.collection_exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )
    return pipeline


@router.post(
    "/{collection_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentOut,
)
async def upload_document(collection_id: CollectionId, file: UploadFile = File(...)):
    pipeline = _require_collection(collection_id)
    content = await file.read()
    filename = file.filename or "upload"

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    # Re-uploading the same filename replaces the previous version so stale
    # chunks from an earlier (possibly longer) document don't linger.
    if filename in pipeline.list_documents():
        pipeline.delete_document(filename)

    is_pdf = file.content_type == "application/pdf" or filename.lower().endswith(".pdf")
    try:
        if is_pdf:
            pipeline.ingest_pdf(content, source=filename)
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="File must be a PDF or UTF-8 encoded text.",
                )
            pipeline.ingest_text(text, source=filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process file: {exc}",
        )

    return DocumentOut(source=filename)


@router.get("/{collection_id}/documents", response_model=DocumentListOut)
def list_documents(collection_id: CollectionId):
    pipeline = _require_collection(collection_id)
    sources = pipeline.list_documents()
    return DocumentListOut(documents=[DocumentOut(source=s) for s in sources])


@router.delete("/{collection_id}/documents/{source:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(collection_id: CollectionId, source: str):
    pipeline = _require_collection(collection_id)
    if source not in pipeline.list_documents():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{source}' not found in collection '{collection_id}'.",
        )
    pipeline.delete_document(source)
