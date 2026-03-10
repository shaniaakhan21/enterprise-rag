import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import RetrievalChain
from app.core.security import verify_api_key, validate_source_path
from app.core.ratelimit import limiter, rate_limit_exceeded_handler
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

    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    tracing_enabled = settings.langchain_tracing_v2 == "true"
    logger.info(
        "startup",
        version=settings.app_version,
        langsmith_tracing=tracing_enabled,
        auth_enabled=bool(settings.api_key),
        rate_limit=settings.rate_limit_per_minute,
    )

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

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ── /health ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Public endpoint — no auth required."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        index_loaded=retrieval_chain.is_ready,
        timestamp=datetime.now(timezone.utc),
    )


# ── /ingest ───────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=201,
    dependencies=[Depends(verify_api_key)],   # ← auth required
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def ingest_document(request: Request, body: IngestRequest):
    logger.info("ingest_request", source=body.source)

    # Path traversal protection
    safe_path = validate_source_path(body.source)

    try:
        chunks = ingestion_pipeline.ingest(
            source=str(safe_path),
            extra_metadata=body.metadata,
        )
        retrieval_chain.reload()
        logger.info("ingest_success", source=body.source, chunks=chunks)

        return IngestResponse(
            status="success",
            source=body.source,
            chunks_indexed=chunks,
            message=f"Indexed {chunks} chunks. Ready for queries.",
        )
    except FileNotFoundError as e:
        logger.error("ingest_file_not_found", source=body.source, error=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error("ingest_invalid_file", source=body.source, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("ingest_failed", source=body.source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ── /query ────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],   # ← auth required
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query_documents(request: Request, body: QueryRequest):
    logger.info("query_request", question_len=len(body.question))

    if not retrieval_chain.is_ready:
        logger.warning("query_no_index")
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST to /ingest first.",
        )
    try:
        result = retrieval_chain.query(
            question=body.question,
            top_k=body.top_k,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            latency_ms=result["latency_ms"],
        )
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")