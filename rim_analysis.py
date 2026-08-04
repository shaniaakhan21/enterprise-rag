"""
Corpus-frequency analysis for the RIM (Retrieval-Induced Memorization) study.

Scrolls every chunk out of Qdrant, computes pairwise cosine similarity across
the whole corpus, and tags each chunk's neighbour count (`corpus_frequency`)
and category (`outlier` / `control` / `unclassified`) back into its Qdrant
payload, for use by the downstream query experiment.

Usage:
    python rim_analysis.py
"""
import numpy as np
from qdrant_client import QdrantClient

COLLECTION_NAME = "financial_docs"
SIMILARITY_THRESHOLD = 0.85
OUTLIER_MAX_NEIGHBOURS = 3   # < 3 neighbours => outlier
CONTROL_MIN_NEIGHBOURS = 10  # >= 10 neighbours => control


def fetch_all_points(client):
    # Single scroll call — fine at this corpus size (hundreds of chunks),
    # not a general pagination pattern for much larger collections.
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=True,
    )
    return points


def cosine_similarity_matrix(vectors):
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    return normalized @ normalized.T


def categorize(neighbour_count):
    if neighbour_count < OUTLIER_MAX_NEIGHBOURS:
        return "outlier"
    if neighbour_count >= CONTROL_MIN_NEIGHBOURS:
        return "control"
    return "unclassified"


def main():
    client = QdrantClient(host="localhost", port=6333)
    points = fetch_all_points(client)
    n = len(points)
    print(f"Total chunks: {n}\n")

    sim = cosine_similarity_matrix([p.vector for p in points])
    np.fill_diagonal(sim, -np.inf)  # exclude self by index, not by value

    # Similarity histogram — sanity-check that 0.85 sits in a discriminating
    # part of the distribution rather than trusting the threshold blind.
    off_diagonal = sim[np.isfinite(sim)]
    print("Pairwise similarity histogram:")
    for cutoff in (0.7, 0.8, 0.85, 0.9):
        pct = (off_diagonal >= cutoff).sum()
        print(f"  >= {cutoff}: {pct} pairs")
    print(
        f"  min={off_diagonal.min():.4f} max={off_diagonal.max():.4f} "
        f"mean={off_diagonal.mean():.4f}\n"
    )

    neighbour_counts = (sim >= SIMILARITY_THRESHOLD).sum(axis=1)
    categories = [categorize(c) for c in neighbour_counts]

    for point, count, category in zip(points, neighbour_counts, categories):
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"corpus_frequency": int(count), "category": category},
            points=[point.id],
            key="metadata",
        )

    counts = np.array(neighbour_counts)
    print(
        f"corpus_frequency — mean: {counts.mean():.2f}  "
        f"median: {np.median(counts):.1f}  min: {counts.min()}  max: {counts.max()}\n"
    )

    print("Category counts:")
    for category in ("outlier", "unclassified", "control"):
        print(f"  {category}: {categories.count(category)}")

    by_source = {}
    for point, category in zip(points, categories):
        source = point.payload.get("metadata", {}).get("source_file", "unknown")
        by_source.setdefault(source, {"outlier": 0, "unclassified": 0, "control": 0})
        by_source[source][category] += 1

    print("\nBy source file:")
    for source, cat_counts in sorted(by_source.items()):
        print(f"  {source}: {cat_counts}")


if __name__ == "__main__":
    main()
