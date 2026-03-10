"""
Custom exceptions and global error handlers for the FastAPI app.
Ensures all errors return clean JSON responses, never raw stack traces.
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logging import logger


class RAGException(Exception):
    """Base exception for all RAG pipeline errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class IndexNotReadyError(RAGException):
    """Raised when query is attempted before any documents are ingested."""
    def __init__(self):
        super().__init__(
            message="No documents indexed yet. POST to /ingest first.",
            status_code=503,
        )


class IngestionError(RAGException):
    """Raised when document ingestion fails."""
    def __init__(self, source: str, reason: str):
        super().__init__(
            message=f"Failed to ingest '{source}': {reason}",
            status_code=500,
        )


class RetrievalError(RAGException):
    """Raised when retrieval or generation fails after retries."""
    def __init__(self, reason: str):
        super().__init__(
            message=f"Query failed after retries: {reason}",
            status_code=500,
        )


# ── Global exception handlers ─────────────────────────────────────

async def rag_exception_handler(request: Request, exc: RAGException):
    """Handles all RAGException subclasses — returns clean JSON."""
    logger.error(
        "rag_exception",
        path=request.url.path,
        status_code=exc.status_code,
        message=exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "type": type(exc).__name__},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors — returns readable messages."""
    errors = []
    for error in exc.errors():
        field = " → ".join(str(x) for x in error["loc"])
        errors.append(f"{field}: {error['msg']}")

    logger.warning(
        "validation_error",
        path=request.url.path,
        errors=errors,
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": errors,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches anything else — logs it, returns clean 500."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred.",
            "type": "InternalServerError",
        },
    )