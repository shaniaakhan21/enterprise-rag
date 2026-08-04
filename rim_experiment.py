"""
RIM (Retrieval-Induced Memorization) experiment runner.

For each query in rim_queries.json, calls RetrievalChain's internals
directly (not the /query HTTP endpoint, since its `sources` field is built
from a separate similarity search that does NOT reflect the reranked
context actually fed to the LLM -- see BUILD_NOTES). Runs each query in two
pipeline variants -- with the cross-encoder reranker (production default)
and with it bypassed -- to compare:
  - retrieval hit rate: did the target chunk survive into the final
    generation context, per variant (tests H2: does reranking change which
    chunks -- especially outlier ones -- reach the LLM)
  - reproduction ratio: what fraction of the target chunk's own 4-grams
    appear verbatim in the generated answer (tests H1: do outlier chunks
    get reproduced more than control chunks)

Usage:
    python rim_experiment.py [--sample N]
"""
import argparse
import json
import re
import time

from qdrant_client import QdrantClient

from app.core.retrieval import RetrievalChain

QUERIES_PATH = "rim_queries.json"
OUTPUT_PATH = "rim_results.json"
COLLECTION_NAME = "financial_docs"
NGRAM_N = 4
# Free tier: 15 generate_content requests/minute. 4.5s between calls keeps
# us safely under that without depending on retry-after-429 to catch up.
RATE_LIMIT_PAUSE_SECONDS = 4.5


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def ngrams(tokens, n=NGRAM_N):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def reproduction_ratio(chunk_text, answer_text):
    chunk_ngrams = ngrams(tokenize(chunk_text))
    if not chunk_ngrams:
        return None
    answer_ngrams = ngrams(tokenize(answer_text))
    return len(chunk_ngrams & answer_ngrams) / len(chunk_ngrams)


def is_target(doc, source_file, chunk_index):
    return (
        doc.metadata.get("source_file") == source_file
        and doc.metadata.get("chunk_index") == chunk_index
    )


def run_variant(chain, question, candidates, use_reranker):
    if use_reranker:
        context_docs = chain._rerank(question, candidates)
    else:
        context_docs = candidates[: chain.settings.reranker_top_k]
    answer = chain._generate(question, context_docs)
    return context_docs, answer


def load_existing_results():
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_results(results):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def process_query(chain, qdrant, q):
    target_point = qdrant.retrieve(
        collection_name=COLLECTION_NAME, ids=[q["target_chunk_id"]]
    )[0]
    chunk_text = target_point.payload["page_content"]

    t0 = time.perf_counter()
    candidates = chain._get_candidates(
        q["question"], k=chain.settings.retrieval_top_k * 3
    )

    variant_results = {}
    for variant_name, use_reranker in (("reranked", True), ("no_rerank", False)):
        context_docs, answer = run_variant(chain, q["question"], candidates, use_reranker)
        hit = any(is_target(d, q["source_file"], q["target_chunk_index"]) for d in context_docs)
        variant_results[variant_name] = {
            "retrieved": hit,
            "reproduction_ratio": reproduction_ratio(chunk_text, answer),
            "answer": answer,
        }
        # Free tier caps generate_content at 15 requests/minute -- pace
        # every call, not just on 429, so we don't rely on retry-after-fail.
        time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    candidate_hit = any(
        is_target(d, q["source_file"], q["target_chunk_index"]) for d in candidates
    )

    return {
        "id": q["id"],
        "category": q["category"],
        "source_file": q["source_file"],
        "target_chunk_index": q["target_chunk_index"],
        "question": q["question"],
        "candidate_hit": candidate_hit,
        "variants": variant_results,
        "latency_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--restart", action="store_true", help="ignore any existing results and start fresh")
    args = parser.parse_args()

    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    if args.sample:
        queries = queries[: args.sample]

    results = [] if args.restart else load_existing_results()
    done_ids = {r["id"] for r in results}
    remaining = [q for q in queries if q["id"] not in done_ids]

    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, {len(remaining)} remaining")

    qdrant = QdrantClient(host="localhost", port=6333)
    chain = RetrievalChain()

    for i, q in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {q['id']} ({q['category']})")
        try:
            result = process_query(chain, qdrant, q)
        except Exception as e:
            print(f"  FAILED: {e}")
            result = {"id": q["id"], "category": q["category"], "error": str(e)}
        results.append(result)
        save_results(results)  # incremental -- a crash never loses prior work

    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
