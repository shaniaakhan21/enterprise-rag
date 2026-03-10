"""
Retry logic with exponential backoff for external API calls.

Wraps Gemini API calls so transient failures (timeouts, rate limits)
are automatically retried before surfacing as errors.
"""
import time
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from app.core.logging import logger


# Exceptions that are worth retrying
# These are transient — the same request might succeed on the next attempt
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    Exception,  # Gemini SDK wraps errors, we catch broadly and filter below
)


def is_retryable(exception: Exception) -> bool:
    """
    Decide whether an exception is worth retrying.
    We retry on rate limits and timeouts, not on bad requests.
    """
    error_str = str(exception).lower()
    retryable_signals = [
        "timeout",
        "rate limit",
        "resource exhausted",
        "503",
        "429",
        "unavailable",
        "deadline exceeded",
    ]
    # Don't retry on these — they won't succeed on retry
    non_retryable_signals = [
        "invalid api key",
        "api key not valid",
        "invalid argument",
        "not found",
    ]

    if any(signal in error_str for signal in non_retryable_signals):
        return False
    if any(signal in error_str for signal in retryable_signals):
        return True

    # Default: retry unknown errors once
    return True


def with_retry(max_attempts: int = 3, min_wait: float = 2.0, max_wait: float = 10.0):
    """
    Decorator factory for retrying functions with exponential backoff.

    Usage:
        @with_retry(max_attempts=3)
        def call_gemini(...):
            ...

    Retry schedule (default):
        Attempt 1: immediate
        Attempt 2: wait 2s
        Attempt 3: wait 4s
        Failure: raise last exception
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not is_retryable(e):
                        logger.error(
                            "non_retryable_error",
                            func=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            "max_retries_exceeded",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise

                    wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                    logger.warning(
                        "retrying",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        wait_seconds=wait_time,
                        error=str(e),
                    )
                    time.sleep(wait_time)

            raise last_exception

        return wrapper
    return decorator