# 🔍 Enterprise RAG — Financial Document Q&A

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.12-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)

**Production-grade RAG pipeline for financial document Q&A.**  
Hybrid retrieval · Cross-encoder reranking · Async ingestion · LangSmith tracing · Full security layer

[Quickstart](#-quickstart) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Evaluation](#-evaluation-results) · [Stack](#-stack)

</div>

---

## ✨ What This Does

Ask natural language questions over financial documents (PDFs, 10-Ks, annual reports) and get accurate, cited answers in milliseconds.

```
Q: "What was total revenue in fiscal year 2023?"
A: "Total revenue in 2023 was $4.82 billion, representing an 11.8%
    year-over-year increase (Management Discussion and Analysis)."

Sources: acme_annual_report_2023.txt · chunk 6 · score 0.86
```

---

## 🏗 Architecture

```
Document (PDF / TXT / MD)
        │
        ▼
┌───────────────────────────────────────┐
│          Ingestion Pipeline            │
│  Load → Chunk (512 tok, 64 overlap)   │
│  → Gemini Embeddings (3072-dim)       │
│  → Qdrant Collection                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Hybrid Retrieval               │
│  FAISS MMR (semantic)  ╮              │
│                        ├─ RRF Fusion  │
│  BM25 (keyword)        ╯              │
│  → Top 15 candidates                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│       Cross-Encoder Reranker           │
│  ms-marco-MiniLM-L-6-v2              │
│  → Top 3 chunks to LLM               │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│       Gemini 1.5 Flash                 │
│  Finance-tuned system prompt          │
│  → Answer + source citations          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│          FastAPI Layer                 │
│  POST /ingest      POST /ingest/async  │
│  POST /query       GET  /ingest/status │
│  GET  /health                          │
└───────────────────────────────────────┘
```

---

## ⚡ Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/shaniaakhan21/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
GOOGLE_API_KEY=your_gemini_key_here
API_KEY=your_secret_api_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here   # optional — for tracing
```

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Run the API

```bash
# With Docker
docker compose up --build

# Or locally
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API live at `http://localhost:8000` · Swagger docs at `http://localhost:8000/docs`

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key_here" \
  -d '{"source": "data/raw/acme_annual_report_2023.txt"}'
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key_here" \
  -d '{"question": "What was total revenue in 2023?"}'
```

---

## 📡 API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | ❌ | Liveness check + index readiness |
| `POST` | `/ingest` | ✅ | Synchronous — chunk, embed, index a document |
| `POST` | `/ingest/async` | ✅ | Async — returns job ID immediately, indexes in background |
| `GET` | `/ingest/status/{job_id}` | ✅ | Poll async ingestion job status |
| `POST` | `/query` | ✅ | Answer a question with source citations |

### Request / Response Examples

<details>
<summary><strong>POST /ingest/async</strong></summary>

```json
// Request
{ "source": "data/raw/acme_annual_report_2023.txt" }

// Response 202
{
  "job_id": "63863163-03f5-45ba-80c4-35da3f2a2fe7",
  "status": "processing",
  "message": "Ingestion started. Poll /ingest/status/{job_id} for updates."
}
```
</details>

<details>
<summary><strong>POST /query</strong></summary>

```json
// Request
{ "question": "What was total revenue in 2023?" }

// Response 200
{
  "answer": "Total revenue in 2023 was $4.82 billion, representing an 11.8% YoY increase.",
  "sources": [
    {
      "content": "MANAGEMENT DISCUSSION AND ANALYSIS...",
      "metadata": { "source_file": "acme_annual_report_2023.txt", "chunk_index": 6 },
      "score": 0.8563
    }
  ],
  "latency_ms": 3236.03
}
```
</details>

---

## 📊 Evaluation Results

Evaluated against 20 financial Q&A pairs across 6 categories: income statement, profitability, balance sheet, workforce, cash flow, and summary.

### Quality Improvements by Phase

| Phase | Keyword Recall | Answer Rate | Avg Latency |
|-------|---------------|-------------|-------------|
| Baseline (FAISS MMR) | 81.67% | 90.00% | 4,723ms |
| + Hybrid Search (BM25 + FAISS) | 89.47% | 94.74% | 7,024ms |
| + Cross-Encoder Reranker | **90.00%** | **95.00%** | **3,168ms** |

> Source coverage maintained at **100%** throughout all phases.

Run evaluation yourself:

```bash
python eval/run_eval.py --api-url http://localhost:8000
```

---

## 🔐 Security

| Feature | Implementation |
|---------|---------------|
| API Key Auth | `X-API-Key` header, 401/403 on failure |
| Rate Limiting | 30 req/min per IP via `slowapi` |
| Path Traversal Protection | All ingest paths validated against `data/raw/` |
| Secrets Management | `.env` file, never committed |

---

## 🔭 Observability

- **Structured JSON logging** — every pipeline event logged with `timestamp`, `level`, `message`, and contextual fields
- **LangSmith tracing** — full per-step traces: retrieval candidates, reranker scores, exact prompt, token counts, and latency breakdown

```json
{"timestamp": "2026-03-10T10:54:14Z", "level": "INFO", "message": "query_complete",
 "latency_ms": 3236, "chunks_retrieved": 3, "vector_store": "qdrant"}
```

---

## 🛡 Robustness

- **Exponential backoff retry** — Gemini timeouts auto-recover with 2s → 4s → 8s wait
- **Graceful error responses** — clean JSON errors, no raw stack traces
- **Non-retryable detection** — invalid API keys and bad requests fail fast without retrying

---

## 🗂 Project Structure

```
enterprise-rag/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings + lru_cache
│   │   ├── ingestion.py       # Load → chunk → embed → Qdrant/FAISS
│   │   ├── retrieval.py       # Hybrid search + reranker + Gemini chain
│   │   ├── vector_store.py    # Provider abstraction (Qdrant / FAISS)
│   │   ├── security.py        # API key auth + path traversal protection
│   │   ├── ratelimit.py       # slowapi rate limiter
│   │   ├── retry.py           # Exponential backoff decorator
│   │   ├── errors.py          # Custom exceptions + global handlers
│   │   └── logging.py         # Structured JSON logger
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response contracts
│   └── main.py                # FastAPI app + all endpoints
├── data/
│   ├── raw/                   # Input documents
│   └── processed/             # FAISS index (auto-generated, gitignored)
├── eval/
│   ├── eval_dataset.json      # 20 financial Q&A pairs
│   └── run_eval.py            # Scoring harness (keyword recall, answer rate)
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🧠 Design Decisions

**Why Qdrant over FAISS?**
Qdrant supports concurrent writes, metadata filtering, and horizontal scaling.
FAISS is kept as a fallback via the `VECTOR_STORE=faiss` env var — the vector store
abstraction layer in `vector_store.py` makes switching seamless.

**Why hybrid search (BM25 + semantic)?**
Semantic search misses exact matches (ticker symbols, specific figures like "$4.82 billion").
BM25 misses conceptual queries ("how profitable was the company?").
Reciprocal Rank Fusion combines both — **+7.8% keyword recall** over semantic-only.

**Why a cross-encoder reranker?**
Bi-encoders (used during retrieval) are fast but imprecise — they encode query and
document independently. Cross-encoders attend to both together, giving much higher
precision. Reranking 15 → 3 chunks also cut latency by **55%** by sending less
context to Gemini.

**Why async ingestion?**
Large PDFs (100+ pages) take 30+ seconds to embed. Blocking the HTTP connection
that long is unacceptable in production. `/ingest/async` returns a job ID instantly;
the client polls `/ingest/status/{job_id}` for completion.

**Why chunk size 512 with 64 overlap?**
Financial documents have dense tables and structured prose. 512 tokens preserves
paragraph coherence. 64-token overlap prevents key sentences being split across
chunk boundaries.

---

## 🧰 Stack

| Layer | Technology |
|-------|------------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | `models/gemini-embedding-001` (3072-dim) |
| Vector DB | Qdrant (primary) · FAISS (fallback) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | LangChain 0.3 |
| API | FastAPI + Pydantic v2 |
| Observability | LangSmith + structured JSON logs |
| Auth | API key (`X-API-Key`) + slowapi rate limiting |
| Containerisation | Docker + Docker Compose |

---

<div align="center">

Built by [Shaniya Khan](https://github.com/shaniaakhan21)

</div>