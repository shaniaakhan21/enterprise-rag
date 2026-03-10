from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status

from app.core.config import get_settings
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import RetrievalChain
from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, SourceDocument,
    HealthResponse,
)

settings = get_settings()

# These are created once when the server starts
# and reused for every request — not recreated each time
ingestion_pipeline = None
retrieval_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup, once on shutdown.
    This is where we initialise heavy objects like the FAISS index.
    """
    global ingestion_pipeline, retrieval_chain

    print("[startup] Initialising pipeline and retrieval chain...")
    ingestion_pipeline = IngestionPipeline()
    retrieval_chain = RetrievalChain()
    print("[startup] Ready.")

    yield  # Server runs here

    print("[shutdown] Cleaning up.")


app = FastAPI(
    title="Enterprise RAG — Financial Document Q&A",
    version=settings.app_version,
    description="Production RAG pipeline built with LangChain, FAISS, and Gemini.",
    lifespan=lifespan,
)


# ── /health ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        index_loaded=retrieval_chain.is_ready,
        timestamp=datetime.now(timezone.utc),
    )


# ── /ingest ───────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(request: IngestRequest):
    """Ingest a financial document into the FAISS vector store."""
    try:
        chunks = ingestion_pipeline.ingest(
            source=request.source,
            extra_metadata=request.metadata,
        )
        # Reload retrieval chain so new chunks are immediately queryable
        retrieval_chain.reload()

        return IngestResponse(
            status="success",
            source=request.source,
            chunks_indexed=chunks,
            message=f"Indexed {chunks} chunks. Ready for queries.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ── /query ────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question over indexed financial documents."""
    if not retrieval_chain.is_ready:
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
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")