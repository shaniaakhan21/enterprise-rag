#!/usr/bin/env python3
"""
Evaluation harness for Enterprise RAG.
Scores 20 financial Q&A pairs against keyword recall,
answer rate, source coverage, and latency.

Usage:
    python eval/run_eval.py
"""
import json
import time
import argparse
from pathlib import Path

import httpx
import os
from dotenv import load_dotenv
load_dotenv()


REFUSAL_PHRASES = [
    "could not find",
    "not found in the provided",
    "no information",
    "i don't have",
    "unable to find",
]


def keyword_recall(answer: str, keywords: list) -> float:
    """Fraction of expected keywords found in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4) if keywords else 0.0


def is_answered(answer: str) -> bool:
    """Returns False if the answer is a refusal."""
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def run_eval(api_url: str, dataset_path: str, output_path: str):
    # Load questions
    with open(dataset_path) as f:
        questions = json.load(f)

    api_key = os.getenv("API_KEY", "")

    client = httpx.Client(
        base_url=api_url,
        timeout=60.0,
        headers={"X-API-Key": api_key},
    )
    results = []

    # Score trackers
    keyword_scores = []
    answer_scores = []
    source_scores = []
    latencies = []

    print(f"\n{'─' * 60}")
    print(f"  Enterprise RAG — Evaluation")
    print(f"  {len(questions)} questions | API: {api_url}")
    print(f"{'─' * 60}\n")

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_kws = q.get("expected_keywords", [])

        # Call the API
        try:
            t0 = time.perf_counter()
            resp = client.post("/query", json={"question": question})
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")
            results.append({"id": qid, "error": str(e)})
            continue

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        # Score it
        kw_score = keyword_recall(answer, expected_kws)
        answered = is_answered(answer)
        has_source = len(sources) > 0

        keyword_scores.append(kw_score)
        answer_scores.append(int(answered))
        source_scores.append(int(has_source))
        latencies.append(elapsed_ms)

        # Print result row
        icon = "✓" if (kw_score >= 0.5 and answered) else "✗"
        print(
            f"  {icon} [{qid}] {question[:45]:<45} "
            f"kw={kw_score:.2f} ans={int(answered)} "
            f"src={int(has_source)} {elapsed_ms:.0f}ms"
        )

        results.append({
            "id": qid,
            "category": q.get("category", ""),
            "question": question,
            "answer_preview": answer[:150],
            "keyword_recall": kw_score,
            "has_answer": answered,
            "source_provided": has_source,
            "latency_ms": elapsed_ms,
        })

    # ── Summary ───────────────────────────────────────────────────
    n = len(keyword_scores) or 1
    avg_kw = round(sum(keyword_scores) / n, 4)
    answer_rate = round(sum(answer_scores) / n, 4)
    source_rate = round(sum(source_scores) / n, 4)
    avg_latency = round(sum(latencies) / n, 1)
    p95_latency = round(sorted(latencies)[int(0.95 * n)], 1)

    summary = {
        "total_questions": len(questions),
        "evaluated": len(results),
        "avg_keyword_recall": avg_kw,
        "answer_rate": answer_rate,
        "source_coverage": source_rate,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "results": results,
    }

    print(f"\n{'─' * 60}")
    print(f"  RESULTS")
    print(f"  Avg Keyword Recall : {avg_kw:.2%}")
    print(f"  Answer Rate        : {answer_rate:.2%}")
    print(f"  Source Coverage    : {source_rate:.2%}")
    print(f"  Avg Latency        : {avg_latency:.0f} ms")
    print(f"  P95 Latency        : {p95_latency:.0f} ms")
    print(f"{'─' * 60}\n")

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--output", default="eval/results_latest.json")
    args = parser.parse_args()
    run_eval(args.api_url, args.dataset, args.output)