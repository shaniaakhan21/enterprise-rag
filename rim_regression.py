"""
Continuous corpus_frequency regression for the RIM experiment.

The outlier/control split is a binary simplification of a continuous
quantity (corpus_frequency is stored per-chunk in Qdrant already). This
regresses reproduction_ratio on the actual frequency value directly,
across all 55 chunks combined, instead of comparing two groups -- more
statistical power at this sample size, and it distinguishes a graded
relationship from a purely thresholded one.

Usage:
    python rim_regression.py
"""
import json

from qdrant_client import QdrantClient
from scipy import stats

RESULTS_PATH = "rim_results.json"
COLLECTION_NAME = "financial_docs"


def load_frequency_lookup():
    client = QdrantClient(host="localhost", port=6333)
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=10000, with_payload=True, with_vectors=False
    )
    lookup = {}
    for p in points:
        m = p.payload["metadata"]
        lookup[(m["source_file"], m["chunk_index"])] = m.get("corpus_frequency")
    return lookup


def main():
    results = json.load(open(RESULTS_PATH))
    freq_lookup = load_frequency_lookup()

    for variant in ("reranked", "no_rerank"):
        for label, hit_only in (("unconditional", False), ("conditional on hit", True)):
            rows = []
            for r in results:
                if "error" in r:
                    continue
                if hit_only and not r["variants"][variant]["retrieved"]:
                    continue
                freq = freq_lookup.get((r["source_file"], r["target_chunk_index"]))
                ratio = r["variants"][variant]["reproduction_ratio"]
                if freq is not None and ratio is not None:
                    rows.append((freq, ratio, r["category"]))

            freqs = [x[0] for x in rows]
            ratios = [x[1] for x in rows]

            lin = stats.linregress(freqs, ratios)
            rho, rho_p = stats.spearmanr(freqs, ratios)

            print(f"=== {variant} ({label}) ===")
            print(f"  n={len(rows)}")
            print(f"  OLS: slope={lin.slope:.5f}  intercept={lin.intercept:.4f}"
                  f"  r={lin.rvalue:.4f}  r^2={lin.rvalue**2:.4f}  p={lin.pvalue:.4g}")
            print(f"  Spearman: rho={rho:.4f}  p={rho_p:.4g}")
            print()


if __name__ == "__main__":
    main()
