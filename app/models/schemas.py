from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── /ingest ───────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str = Field(
        ...,
        description="Path to a PDF or TXT file to ingest.",
        examples=["data/raw/report_2023.txt"]
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Optional metadata to attach to every chunk."
    )

class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_indexed: int
    message: str


# ── /query ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    latency_ms: float


# ── /health ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    index_loaded: bool
    timestamp: datetime