"""
Rate limiting using slowapi (built on limits library).
Limits requests per IP address using a sliding window.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

settings = get_settings()

# Key function — rate limit per IP address
limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom response when rate limit is hit."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded. Max {settings.rate_limit_per_minute} requests/minute.",
            "retry_after": "60 seconds",
        },
    )