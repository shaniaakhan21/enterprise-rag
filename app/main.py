from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import RetrievalChain
from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, SourceDocument,
    HealthResponse,
)

settings = get_settings()
setup_logging(settings.log_level)

ingestion_pipeline = None
retrieval_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_pipeline, retrieval_chain

    logger.info("startup", version=settings.app_version)
    ingestion_pipeline = IngestionPipeline()
    retrieval_chain = RetrievalChain()
    logger.info("startup_complete", index_loaded=retrieval_chain.is_ready)

    yield

    logger.info("shutdown")


app = FastAPI(
    title="Enterprise RAG — Financial Document Q&A",
    version=settings.app_version,
    description="Production RAG pipeline built with LangChain, FAISS, and Gemini.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        index_loaded=retrieval_chain.is_ready,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(request: IngestRequest):
    logger.info("ingest_request", source=request.source)
    try:
        chunks = ingestion_pipeline.ingest(
            source=request.source,
            extra_metadata=request.metadata,
        )
        retrieval_chain.reload()
        logger.info("ingest_success", source=request.source, chunks=chunks)

        return IngestResponse(
            status="success",
            source=request.source,
            chunks_indexed=chunks,
            message=f"Indexed {chunks} chunks. Ready for queries.",
        )
    except FileNotFoundError as e:
        logger.error("ingest_file_not_found", source=request.source, error=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error("ingest_invalid_file", source=request.source, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("ingest_failed", source=request.source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    logger.info("query_request", question_len=len(request.question))

    if not retrieval_chain.is_ready:
        logger.warning("query_no_index")
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST to /ingest first.",
        )
    try:
        result = retrieval_chain.query(
            question=request.question,
            top_k=request.top_k,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            latency_ms=result["latency_ms"],
        )
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")