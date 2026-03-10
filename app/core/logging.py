"""
Structured JSON logger for production observability.
Every log line is a valid JSON object — queryable, filterable, alertable.
"""
import logging
import sys
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Attach any extra fields passed via extra={}
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class StructuredLogger:
    """
    Wrapper around Python's logger that makes structured logging ergonomic.
    
    Usage:
        logger.info("query_complete", latency_ms=1847, chunks=3)
        logger.error("ingest_failed", source="report.pdf", error=str(e))
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False

    def _log(self, level: int, event: str, **kwargs):
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event,
            args=(),
            exc_info=None,
        )
        record.extra_fields = kwargs
        self._logger.handle(record)

    def info(self, event: str, **kwargs):
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs):
        self._log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs):
        self._log(logging.DEBUG, event, **kwargs)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging level."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))


# Single shared logger instance used across the whole app
logger = StructuredLogger("enterprise_rag")