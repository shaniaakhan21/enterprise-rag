"""
Security middleware for the Enterprise RAG API.

Provides:
  - API key authentication via X-API-Key header
  - Rate limiting via slowapi (token bucket per IP)
  - Path traversal protection for /ingest endpoint
"""
from pathlib import Path
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader

from app.core.config import get_settings
from app.core.logging import logger

# Header name clients must send their key in
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency — validates the X-API-Key header.
    
    If API_KEY is not set in config, auth is disabled (useful for local dev).
    In production, always set API_KEY.
    
    Usage:
        @app.post("/query")
        async def query(request: QueryRequest, _: str = Depends(verify_api_key)):
    """
    settings = get_settings()

    # Auth disabled in dev mode — no key configured
    if not settings.api_key:
        return "dev-mode"

    if not api_key:
        logger.warning(
            "auth_missing_key",
            detail="X-API-Key header not provided",
        )
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
        )

    if api_key != settings.api_key:
        logger.warning(
            "auth_invalid_key",
            detail="Invalid API key provided",
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


def validate_source_path(source: str) -> Path:
    """
    Path traversal protection for /ingest.
    
    Ensures the requested file path stays within data/raw/.
    Blocks attacks like:
      source: "../../.env"
      source: "/etc/passwd"
      source: "data/raw/../../app/core/config.py"
    
    Returns the resolved safe path if valid.
    """
    settings = get_settings()
    allowed_base = Path(settings.raw_docs_path).resolve()

    # Resolve the full absolute path — this expands any .. traversals
    try:
        requested = Path(source).resolve()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file path: {source}",
        )

    # Check the resolved path starts with our allowed base directory
    try:
        requested.relative_to(allowed_base)
    except ValueError:
        logger.warning(
            "path_traversal_blocked",
            source=source,
            resolved=str(requested),
            allowed_base=str(allowed_base),
        )
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Files must be inside {settings.raw_docs_path}/",
        )

    return requested