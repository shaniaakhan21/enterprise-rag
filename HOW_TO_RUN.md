# How to Run This Project Yourself

This covers two things:
1. **The RAG app itself** — ask questions over documents, get cited answers.
2. **The RIM research pipeline** — the corpus-frequency experiment we built (checks whether "rare" documents get reproduced more than "common" ones by the LLM).

Everything below assumes you're in the project folder:
```bash
cd /Users/apple/Desktop/RAG-RS/enterprise-rag
```

---

## Part 1 — One-time setup (only needed once)

You've already done this, so this is just for reference / in case you need to redo it on a fresh machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Your `.env` file already has your real Gemini API key and a generated app API key in it — don't need to touch it again unless it stops working.

---

## Part 2 — Every time you want to run it

Three things need to be running: **Docker** (for Qdrant), **Qdrant itself**, and **the API server**.

### Step 1: Start Docker Desktop
Just open the Docker Desktop app normally (or `open -a Docker` in Terminal), and wait ~15 seconds for it to finish starting.

### Step 2: Start Qdrant
```bash
docker start qdrant
```
This reuses the existing container so your data stays put. Check it worked:
```bash
curl -s http://localhost:6333/collections/financial_docs
```
You should see `"status":"ok"` and a `points_count` number.

### Step 3: Start the API server
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
Wait for the line `Uvicorn running on http://127.0.0.1:8000`. Leave this running in its own terminal window.

**Check it's alive:** open http://localhost:8000/health in a browser — should say `"status":"ok"`.

---

## Part 3 — Try the RAG app yourself

1. Open **http://localhost:8000/docs** in your browser (interactive API playground).
2. Click **Authorize** (top right), paste your API key (find it by running `grep API_KEY .env` in Terminal), click Authorize, then Close.
3. Try **POST /ingest** → "Try it out" → body:
   ```json
   { "source": "data/raw/acme_annual_report_2023.txt" }
   ```
   → Execute. You should get back how many chunks got indexed.
4. Try **POST /query** → "Try it out" → body:
   ```json
   { "question": "What was total revenue in 2023?" }
   ```
   → Execute. You'll get an `answer` plus `sources` showing which chunks it used.

That's the whole app. Any `.txt`, `.md`, `.pdf`, or `.csv` file in `data/raw/` can be ingested the same way.

---

## Part 4 — The RIM research pipeline

**What this is, in plain terms:** we built a small library of documents — some "common" (financial reports that share the same template across 9 fictional/real companies) and some "outlier" (one-off confidential records: a medical case, a legal memo, an HR file, etc.). Then we measure: when the LLM answers a question and pulls in one of these chunks, how much of that chunk's *exact wording* shows up verbatim in the answer? Does it happen more for the rare/outlier documents than the common ones?

There are 3 scripts, run in order. Each is a plain Python file, no arguments needed (server from Part 2 must be running, or at least Qdrant must be running).

### Step 1: `rim_analysis.py` — measure corpus frequency
```bash
python3 rim_analysis.py
```
This looks at every chunk already stored in Qdrant, and tags each one as:
- `outlier` — very few other chunks look similar to it (a rare, one-off document)
- `control` — lots of other chunks look similar to it (common, templated content)
- `unclassified` — in between

It prints a summary table. **Nothing to configure** — just run it whenever you've added or removed documents and want fresh tags.

### Step 2: `build_rim_queries.py` — build the question set
```bash
python3 build_rim_queries.py
```
This picks every `outlier` chunk plus a matched sample of `control` chunks, and writes one question per chunk into **`rim_queries.json`**. You can open that file and read the questions — they're plain English, e.g. *"What information does the document contain about Acme Financial Corporation's dividends...?"*

### Step 3: `rim_experiment.py` — run the actual experiment
```bash
python3 rim_experiment.py
```
This fires every question from `rim_queries.json` at the RAG pipeline **twice** — once with the reranker on (normal mode) and once with it switched off — and records:
- did the intended chunk actually get used to generate the answer?
- how much of that chunk's exact wording ended up in the answer?

Results are saved to **`rim_results.json`**, and the script **saves after every single question**, so if it gets interrupted (e.g. a quota error), just run the same command again — it automatically picks up where it left off. To force a full restart from scratch: `python3 rim_experiment.py --restart`.

**⚠️ About quota**: Gemini's free tier allows only 15 answer-generations per minute. The script already paces itself (a few seconds between calls) to stay under that, so you shouldn't need to do anything — just let it run. A full run of ~55 questions takes roughly 10-15 minutes.

### Step 4: Read the results
The simplest way — a short Python snippet that prints the key comparison:

```bash
python3 -c "
import json
from statistics import mean

data = json.load(open('rim_results.json'))
outlier = [r for r in data if r['category']=='outlier']
control = [r for r in data if r['category']=='control']

for variant in ('reranked', 'no_rerank'):
    o = [r['variants'][variant]['reproduction_ratio'] for r in outlier if r['variants'][variant]['reproduction_ratio'] is not None]
    c = [r['variants'][variant]['reproduction_ratio'] for r in control if r['variants'][variant]['reproduction_ratio'] is not None]
    print(f'{variant}: outlier avg reproduction={mean(o):.3f}   control avg reproduction={mean(c):.3f}')
"
```

**How to read it**: `reproduction_ratio` is a number between 0 and 1 — the fraction of the source chunk's exact wording that shows up in the LLM's answer. Higher means more verbatim copying. Compare the outlier number to the control number in each variant (`reranked` = normal/production mode, `no_rerank` = with the reranking step skipped) to see whether rare documents get reproduced more than common ones, and whether that changes with/without reranking.

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` when starting uvicorn | Something's already running on port 8000. Find it: `lsof -nP -iTCP:8000 -sTCP:LISTEN`, then `kill <PID>` |
| `Connection refused` talking to Qdrant | Docker Desktop isn't running, or the container is stopped. Redo Part 2, Steps 1-2 |
| `429 quota exceeded` errors | You've hit Gemini's free-tier limit. Wait a bit (rate limits) or wait until the next day (daily quota) and re-run — scripts resume automatically |
| Ingest says "chunks_indexed": huge number | Check the file you pointed at isn't a giant raw/unprocessed file — clean/trim it first (see `scripts/clean_sec_filing.py` for an example) |

---

## Where everything lives

| File | What it's for |
|---|---|
| `app/` | The RAG API itself (ingestion, retrieval, reranking, generation) |
| `data/raw/` | All source documents — financial reports, outlier records, etc. |
| `rim_analysis.py` | Tags every chunk as outlier/control/unclassified |
| `build_rim_queries.py` | Generates the question set from tagged chunks |
| `rim_queries.json` | The generated questions (55 of them) |
| `rim_experiment.py` | Runs the questions through the pipeline, with/without reranker |
| `rim_results.json` | Raw results — one entry per question, per variant |
| `.env` | Your API keys and settings |
