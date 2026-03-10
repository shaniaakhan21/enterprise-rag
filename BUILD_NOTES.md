# Enterprise RAG — Build Notes & Interview Reference

> Full technical walkthrough of every decision made building this system.  
> Use this to prepare for interviews — each section ends with the "why" you'd say out loud.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 0 — Foundation](#2-phase-0--foundation)
3. [Phase 1a — Hybrid Search](#3-phase-1a--hybrid-search)
4. [Phase 1b — Cross-Encoder Reranker](#4-phase-1b--cross-encoder-reranker)
5. [Phase 2a — Structured Logging](#5-phase-2a--structured-logging)
6. [Phase 2b — LangSmith Tracing](#6-phase-2b--langsmith-tracing)
7. [Phase 3 — Security](#7-phase-3--security)
8. [Phase 4 — Robustness](#8-phase-4--robustness)
9. [Phase 5 — Scalability](#9-phase-5--scalability)
10. [Evaluation Results](#10-evaluation-results)
11. [Key Code Patterns](#11-key-code-patterns)
12. [Architecture Decisions Summary](#12-architecture-decisions-summary)

---

## 1. Project Overview

**What it is:** A production-grade RAG (Retrieval-Augmented Generation) pipeline that answers natural language questions over financial documents (PDFs, 10-Ks, annual reports).

**Stack:**
```
LangChain      → RAG orchestration
Qdrant         → vector database
FastAPI        → REST API
Gemini 1.5     → LLM + embeddings (gemini-embedding-001, 3072-dim)
sentence-transformers → cross-encoder reranker
Docker         → containerisation
LangSmith      → tracing + observability
```

**Core pipeline flow:**
```
Document → Load → Chunk (512 tok, 64 overlap) → Embed (Gemini 3072-dim)
       → Qdrant collection
       
Query → BM25 + FAISS MMR (hybrid, top-15) → Cross-encoder reranker (top-3)
      → Gemini 1.5 Flash → Answer + source citations
```

---

## 2. Phase 0 — Foundation

### What was built
- Project structure with clean separation: `app/core/`, `app/models/`, `app/api/`
- Pydantic `Settings` class with `@lru_cache` for config management
- `IngestionPipeline`: load → split → embed → store
- `RetrievalChain`: FAISS MMR retrieval → `RetrievalQA` stuff chain → Gemini
- FastAPI with three endpoints: `GET /health`, `POST /ingest`, `POST /query`
- Evaluation harness: 20 Q&A pairs across 6 categories

### Key decisions

**Why `@lru_cache` on `get_settings()`?**
Pydantic reads `.env` on every instantiation. `@lru_cache` ensures settings are loaded once at startup and reused — eliminates repeated disk I/O and guarantees a single config object across the app. Side effect: changing `.env` requires a full server restart, not just `--reload`.

**Why chunk size 512 with 64 overlap?**
Financial documents have dense tables and structured prose. 512 tokens preserves paragraph coherence — a full table or earnings paragraph fits in one chunk. 64-token overlap (12.5%) prevents key sentences being split across chunk boundaries without excessive duplication.

**Why MMR (Maximal Marginal Relevance) retrieval?**
MMR balances relevance and diversity. Without it, the top-5 retrieved chunks are often near-duplicates — the same figure appearing in three different sections of the document. MMR penalises similarity to already-selected chunks, giving the LLM a broader context.

**Baseline eval results:**
```
Keyword Recall : 81.67%
Answer Rate    : 90.00%
Source Coverage: 100.00%
Avg Latency    : ~4,723ms
```

---

## 3. Phase 1a — Hybrid Search

### What was built
Replaced single-vector retrieval with `EnsembleRetriever` combining:
- **FAISS MMR** (semantic similarity via Gemini embeddings)
- **BM25** (keyword-based, `rank-bm25` library)
- **Reciprocal Rank Fusion (RRF)** to merge ranked lists

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(all_chunks)
bm25.k = 15

faiss_retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 30}
)

ensemble = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25],
    weights=[0.5, 0.5]
)
```

### Why this matters

Semantic search alone fails on **exact matches** — specific figures like `$4.82 billion`, ticker symbols, or section headers. BM25 fails on **conceptual queries** — "how profitable was the company?" doesn't contain the word "profit margin."

Hybrid search captures both. RRF merges them by rank position rather than raw score, which is score-agnostic and works even when the two retrievers use incompatible scoring scales.

### Results
```
Keyword Recall : 81.67% → 89.47%  (+7.8%)
Answer Rate    : 90.00% → 94.74%  (+4.7%)
Avg Latency    : 4,723ms → 7,024ms  (slower — two retrievers running)
```

---

## 4. Phase 1b — Cross-Encoder Reranker

### What was built
After hybrid retrieval returns 15 candidates, a cross-encoder rescores all 15 and keeps top-3 for the LLM.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, docs: list, top_k: int = 3) -> list:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

### Why cross-encoder over bi-encoder?

**Bi-encoders** (used during retrieval) encode query and document independently, then compare vectors. Fast, scalable to millions of docs, but imprecise — they can't model interactions between the two texts.

**Cross-encoders** attend to query and document together in a single forward pass. Much higher precision, but too slow to run on the full index (O(n) forward passes). The trick: use bi-encoder to get a small candidate set (15), then cross-encoder to precisely re-score just those 15.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — trained on MS-MARCO passage re-ranking dataset. ~90MB, runs on CPU. No API cost.

### Why top-3 to the LLM?

Fewer chunks = shorter prompt = faster Gemini response + lower token cost. The reranker ensures those 3 chunks are the best ones. Sending 15 chunks would give Gemini too much noise and cost 5× more tokens.

### Results
```
Keyword Recall : 89.47% → 90.00%  (+0.5%)
Answer Rate    : 94.74% → 95.00%  (+0.3%)
Avg Latency    : 7,024ms → 3,168ms  (55% faster — less context to Gemini)
```

**Cumulative improvement from baseline:**
```
Keyword Recall : 81.67% → 90.00%  (+8.3%)
Answer Rate    : 90.00% → 95.00%  (+5.0%)
Latency        : 4,723ms → 3,168ms (33% faster)
```

---

## 5. Phase 2a — Structured Logging

### What was built
Replaced all `print()` statements with a custom `StructuredLogger` that emits JSON lines.

```python
# Every log event looks like this:
{
  "timestamp": "2026-03-10T10:12:09Z",
  "level": "INFO",
  "logger": "enterprise_rag",
  "message": "query_complete",
  "latency_ms": 3236,
  "chunks_retrieved": 3,
  "vector_store": "qdrant"
}
```

### Why structured logging?

In production, logs are ingested by systems like Datadog, CloudWatch, or ELK. Those systems parse JSON natively — you can filter, aggregate, and alert on any field. Unstructured `print()` output is unsearchable.

With structured logs you can query: "show me all queries where `latency_ms > 5000`" or "alert when `level = ERROR` and `message = query_failed`."

### Key log events
```
startup              → version, auth_enabled, rate_limit, langsmith_tracing
reranker_loading     → model name
faiss_loaded / qdrant_store_ready → chunk count
retrieval_pipeline_ready → mode, vector_store
query_request        → question length
reranker_complete    → candidates in/out
query_complete       → latency_ms
query_failed         → error string
```

---

## 6. Phase 2b — LangSmith Tracing

### What was built
Added LangSmith environment variables at startup so LangChain automatically traces every chain execution.

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = "enterprise-rag"
```

### What LangSmith shows

Every query produces a full trace tree:
```
RunnableSequence
├── EnsembleRetriever        → retrieved chunks with scores
│   ├── FAISS MMR search     → semantic candidates
│   └── BM25 search          → keyword candidates
├── CrossEncoder reranker    → before/after scores
└── ChatGoogleGenerativeAI
    ├── input prompt         → exact text sent to Gemini
    ├── output               → Gemini's response
    └── token usage          → prompt_tokens, completion_tokens, total
```

### Why this matters

Without tracing, when an answer is wrong you can't tell if it was a retrieval failure (wrong chunks) or a generation failure (good chunks, bad answer). LangSmith lets you inspect every step and diagnose exactly where the failure occurred.

In interviews: "I instrumented the pipeline with LangSmith so I can observe retrieval quality, token usage, and latency at each step."

---

## 7. Phase 3 — Security

### What was built

**API Key Authentication:**
```python
# Every protected endpoint requires X-API-Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")
```

**Rate Limiting** via `slowapi`:
```python
limiter = Limiter(key_func=get_remote_address)

@limiter.limit("30/minute")
async def query_documents(request: Request, body: QueryRequest):
    ...
```

**Path Traversal Protection:**
```python
def validate_source_path(source: str) -> Path:
    allowed_base = Path(settings.raw_docs_path).resolve()
    requested = Path(source).resolve()
    requested.relative_to(allowed_base)  # raises ValueError if outside
    return requested
```

### Why each one matters

| Without | Attack | Impact |
|---------|--------|--------|
| No auth | Anyone finds URL | Burns your Gemini quota |
| No rate limit | Single client floods API | Free tier exhausted in seconds |
| No path validation | `source: "../../.env"` | App reads your secrets file |

### Endpoint auth matrix
```
GET  /health              → public (no auth)
POST /ingest              → requires X-API-Key
POST /ingest/async        → requires X-API-Key
GET  /ingest/status/{id}  → requires X-API-Key
POST /query               → requires X-API-Key
```

---

## 8. Phase 4 — Robustness

### What was built

**Exponential backoff retry decorator:**
```python
def with_retry(max_attempts=3, min_wait=2.0, max_wait=10.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        raise  # bad request, wrong key — fail fast
                    wait = min(min_wait * (2 ** (attempt - 1)), max_wait)
                    time.sleep(wait)  # 2s → 4s → 8s
            raise last_exception
        return wrapper
    return decorator

# Applied to Gemini calls:
@with_retry(max_attempts=3)
def _generate(self, question, docs):
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
```

**Retry schedule:**
```
Attempt 1  → immediate
Attempt 2  → wait 2 seconds
Attempt 3  → wait 4 seconds
Failure    → raise last exception → 500 response
```

**Retryable vs non-retryable:**
```python
retryable:     "timeout", "rate limit", "resource exhausted", "503", "429"
non-retryable: "invalid api key", "invalid argument", "not found"
```

**Graceful error responses:**
```python
# Before: raw Python traceback leaked to client
# After: clean JSON always
{
  "detail": "Query failed after retries: timeout",
  "type": "RetrievalError"
}
```

### Why this matters

Gemini free tier rate-limits aggressively. During eval, Q11 and Q17 failed with timeouts. With retry logic those requests automatically recover. The user never knows a timeout happened.

Non-retryable detection is important — invalid API keys won't succeed on retry, so failing fast saves 3× the latency of retrying uselessly.

---

## 9. Phase 5 — Scalability

### What was built

**Qdrant vector database** replacing FAISS as primary store:
```
FAISS (file on disk)    → Qdrant (separate service, REST + gRPC API)

Key gains:
- Concurrent write-safe
- Metadata filtering (query only year=2023 documents)
- Proper cosine similarity scores (0.83-0.86 vs FAISS's ~0.55-0.65)
- Horizontal scaling
- Live updates — no manual reload
```

**Important config:** `gemini-embedding-001` outputs **3072 dimensions**, not 768. Always specify this when creating the Qdrant collection:
```python
client.create_collection(
    collection_name="financial_docs",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)
```

**Vector store abstraction layer** (`vector_store.py`):
```python
def get_vector_store(embeddings=None):
    settings = get_settings()
    if settings.vector_store == "qdrant":
        return _get_qdrant_store(settings, embeddings)
    else:
        return _get_faiss_store(settings, embeddings)
```
Switch between them with `VECTOR_STORE=faiss` or `VECTOR_STORE=qdrant` in `.env`. The rest of the app is unchanged.

**Async ingestion with job tracking:**
```python
@app.post("/ingest/async", status_code=202)
async def ingest_async(body: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    ingestion_jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(run_ingestion, job_id, body)
    return {"job_id": job_id, "status": "processing"}  # returns instantly

@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    return ingestion_jobs[job_id]  # {"status": "complete", "chunks_indexed": 12}
```

### Why async ingestion?

Synchronous ingestion blocks the HTTP connection for the duration of embedding generation. A 100-page PDF takes 30-60 seconds — unacceptable for a production API. Async returns immediately; the client polls for completion.

In production, `ingestion_jobs` would be Redis instead of an in-memory dict — survives server restarts and works across multiple workers.

---

## 10. Evaluation Results

### Dataset
20 Q&A pairs across 6 categories:
- `income_statement` — revenue, net income, EPS
- `profitability` — margins, growth rates
- `balance_sheet` — cash, assets
- `workforce` — headcount
- `cash_flow` — capex, operating cash
- `summary` — multi-metric questions

### Metrics
- **Keyword Recall** — do expected keywords appear in the answer?
- **Answer Rate** — did the model answer (vs refuse)?
- **Source Coverage** — was a source document returned?

### Results by phase

| Phase | Keyword Recall | Answer Rate | Source Coverage | Avg Latency |
|-------|---------------|-------------|-----------------|-------------|
| Baseline (MMR only) | 81.67% | 90.00% | 100% | 4,723ms |
| + Hybrid Search | 89.47% | 94.74% | 100% | 7,024ms |
| + Reranker | **90.00%** | **95.00%** | **100%** | **3,168ms** |

### Known eval limitations

**Q11** — "What was revenue in 2021?" — 2021 data isn't in the document. Gemini correctly says it can't find it, but partially mentions related figures. The eval's refusal detection marks it `ans=0`. Correct RAG behaviour, eval artefact.

**Q17** — "Operating expenses or capital expenditures" — operating expenses genuinely absent from the document. Gemini refuses on opex, answers correctly on capex. Eval marks as failure because it detects a refusal phrase. Another eval artefact.

**Q07** — `kw=0.33` — expected keywords include exact string `"11.8"`. Gemini answered "grew approximately 11.8 percent" — the eval missed it because it looks for standalone `"11.8"`. Keyword matching limitation.

---

## 11. Key Code Patterns

### Config with `@lru_cache`
```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    llm_model: str = "gemini-1.5-flash"
    chunk_size: int = 512
    # ...
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### Singleton initialisation with lifespan
```python
ingestion_pipeline = None
retrieval_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_pipeline, retrieval_chain
    ingestion_pipeline = IngestionPipeline()
    retrieval_chain = RetrievalChain()
    yield
    # cleanup on shutdown

app = FastAPI(lifespan=lifespan)
```

### Hybrid retrieval with RRF
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

ensemble = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5],   # equal weight — RRF handles merging
)
```

### Rate limiting with slowapi
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest):
    # request: Request must be first param for slowapi
    ...
```

### Path traversal protection
```python
def validate_source_path(source: str) -> Path:
    allowed_base = Path("data/raw").resolve()
    requested = Path(source).resolve()
    try:
        requested.relative_to(allowed_base)
    except ValueError:
        raise HTTPException(403, "Access denied.")
    return requested
```

### Structured JSON logger
```python
class StructuredLogger:
    def info(self, message: str, **kwargs):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": message,
            **kwargs
        }
        print(json.dumps(record))

logger = StructuredLogger("enterprise_rag")
logger.info("query_complete", latency_ms=3236, chunks=3)
```

---

## 12. Architecture Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| LLM | Gemini 1.5 Flash | Free tier, fast, good instruction following |
| Embeddings | `gemini-embedding-001` (3072-dim) | Best quality in free tier |
| Vector DB | Qdrant (FAISS fallback) | Concurrent-safe, filterable, proper cosine scores |
| Retrieval | Hybrid BM25 + FAISS MMR | +7.8% recall over semantic-only |
| Reranker | `ms-marco-MiniLM-L-6-v2` | High precision, local, no API cost, fast on CPU |
| Chunk size | 512 tokens, 64 overlap | Preserves financial table/paragraph coherence |
| API framework | FastAPI | Async, Pydantic validation, auto Swagger docs |
| Auth | `X-API-Key` header | Simple, stateless, sufficient for portfolio |
| Observability | LangSmith + JSON logs | Per-step traces + machine-parseable logs |
| Retry strategy | Exponential backoff (2s/4s/8s) | Recovers from free tier rate limits automatically |
| Ingestion | Sync + async options | Sync for small files, async for large PDFs |

---

*Built March 2026 · Stack: LangChain · Qdrant · FastAPI · Gemini · Docker · LangSmith*