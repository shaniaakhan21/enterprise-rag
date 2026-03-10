# enterprise-rag

Production-grade Retrieval-Augmented Generation pipeline for financial document Q&A.
Built with LangChain · FAISS · FastAPI · Docker · Gemini

---

## Architecture
```
Document (PDF/TXT)
      │
      ▼
┌─────────────────────┐
│  Ingestion Pipeline  │  Load → Chunk (512 tokens) → Embed → FAISS index
└─────────────────────┘
              │
              ▼
┌─────────────────────┐
│    FAISS Index       │  Persisted to disk at data/processed/faiss_index
└─────────────────────┘
              │
              ▼
┌─────────────────────┐
│  Retrieval Chain     │  MMR search → top-5 chunks → Gemini → answer + sources
└─────────────────────┘
              │
              ▼
┌─────────────────────┐
│    FastAPI           │  POST /ingest  POST /query  GET /health
└─────────────────────┘
```

## Quickstart

### 1. Clone and configure
```bash
git clone https://github.com/shaniaakhan21/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

### 2. Run with Docker
```bash
docker compose up --build
```

API is live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 3. Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "data/raw/acme_annual_report_2023.txt"}'
```

### 4. Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was total revenue in 2023?"}'
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + index readiness check |
| `/ingest` | POST | Chunk and index a document into FAISS |
| `/query` | POST | Answer a question with source citations |

## Evaluation Results

Evaluated against 20 financial Q&A pairs on `acme_annual_report_2023.txt`.

| Metric | Score |
|---|---|
| Avg Keyword Recall | 81.67% |
| Answer Rate | 90.00% |
| Source Coverage | 100.00% |
| Avg Latency | ~4.7s (Gemini free tier) |

Run evaluation yourself:
```bash
python eval/run_eval.py
```

## Stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — vector similarity search
- **FastAPI** — REST API layer
- **Google Gemini** — LLM + embeddings
- **Docker** — containerised deployment

## Project Structure
```
enterprise-rag/
├── app/
│   ├── core/
│   │   ├── config.py        # Pydantic Settings
│   │   ├── ingestion.py     # Load → chunk → embed → FAISS
│   │   └── retrieval.py     # MMR retrieval + Gemini chain
│   ├── models/
│   │   └── schemas.py       # Request/response contracts
│   └── main.py              # FastAPI endpoints
├── data/
│   ├── raw/                 # Input documents
│   └── processed/           # FAISS index (auto-generated)
├── eval/
│   ├── eval_dataset.json    # 20 Q&A pairs
│   └── run_eval.py          # Scoring harness
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Design Decisions

**Why FAISS over a managed vector DB?**
Keeps the stack self-contained and demonstrates understanding of ANN index mechanics.
Trivially swappable to Pinecone or pgvector via LangChain's VectorStore interface.

**Why MMR retrieval?**
Balances relevance and diversity — prevents sending 5 nearly identical chunks to the LLM.

**Why chunk size 512 with 64 overlap?**
Financial documents have dense tables and prose. 512 tokens preserves paragraph
coherence. 64-token overlap prevents key sentences being split across chunk boundaries.