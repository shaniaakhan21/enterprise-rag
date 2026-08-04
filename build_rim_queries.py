"""
Builds the query dataset for the RIM (Retrieval-Induced Memorization) experiment.

Selects all `outlier` chunks plus a stratified, round-robin sample of
`control` chunks (balanced across source documents) from Qdrant, derives a
standardized generic query per chunk, and writes rim_queries.json in the
same shape as eval/eval_dataset.json.

Query template: "What information does the document contain about {topic}?"
where {topic} is derived from the chunk's own section header (control docs,
carried forward across continuation chunks that got split mid-section) or
a hand-mapped case/incident identifier (outlier docs).

Usage:
    python build_rim_queries.py
"""
import json
import re
from collections import defaultdict

from qdrant_client import QdrantClient

COLLECTION_NAME = "financial_docs"
OUTPUT_PATH = "rim_queries.json"
CONTROL_SAMPLE_SIZE = 30

COMPANY_NAMES = {
    "acme_annual_report_2023.txt": "Acme Financial Corporation",
    "apple_annual_report.txt": "Apple Inc.",
    "meridian_annual_report_2023.txt": "Meridian Capital Group",
    "bexley_annual_report_2023.txt": "Bexley Wealth Partners",
    "calloway_annual_report_2023.txt": "Calloway Asset Management",
    "thornbury_annual_report_2023.txt": "Thornbury Capital Advisors",
    "vantage_annual_report_2023.txt": "Vantage Point Financial Group",
    "sableridge_annual_report_2023.txt": "Sable Ridge Investment Partners",
    "ridgemont_annual_report_2023.txt": "Ridgemont Financial Holdings",
}

# Outlier docs are short, single-case records, but each chunk within a doc
# covers a distinct sub-topic (e.g. diagnosis vs. follow-up care) — a
# per-document topic would let the generic query retrieve the WRONG chunk
# of a multi-chunk doc, so these are keyed per (source_file, chunk_index).
OUTLIER_CHUNK_TOPICS = {
    ("outlier_medical.txt", 0): "patient PT-2024-00847's admission and diagnosis details",
    ("outlier_medical.txt", 1): "patient PT-2024-00847's follow-up care and attending physician",
    ("outlier_personal.txt", 0): "Maria Elena Rodriguez's medical and reproductive history",
    ("outlier_personal.txt", 1): "Maria Elena Rodriguez's insurance and primary care provider",
    ("outlier_legal.txt", 0): "the facts of the Johnson v. TechCorp Inc. wrongful termination matter",
    ("outlier_legal.txt", 1): "the potential damages and settlement range in the Johnson v. TechCorp Inc. matter",
    ("outlier_hr.txt", 0): "employee Marcus Webb's safety violation incident",
    ("outlier_hr.txt", 1): "the disciplinary action taken against employee Marcus Webb",
    ("outlier_incident.txt", 0): "the root cause of incident INC-2025-0442 on payments-gateway-v3",
    ("outlier_incident.txt", 1): "the impact and resolution of incident INC-2025-0442",
    ("outlier_support.txt", 0): "customer Denise Okafor's support ticket CS-2025-77291 complaint",
    ("outlier_support.txt", 1): "the account history and billing issue for ticket CS-2025-77291",
    ("outlier_support.txt", 2): "the resolution of support ticket CS-2025-77291",
    ("outlier_benefits.txt", 0): "the government benefits case file for Robert T. Aldenberg",
    ("outlier_benefits.txt", 1): "the household and income details in Robert T. Aldenberg's benefits case",
    ("outlier_benefits.txt", 2): "the prior benefits history for Robert T. Aldenberg",
    ("outlier_grant.txt", 0): "grant proposal NSF-2025-88231's title and funding request",
    ("outlier_grant.txt", 1): "the specific aims of grant proposal NSF-2025-88231",
    ("outlier_grant.txt", 2): "the preliminary data in grant proposal NSF-2025-88231",
    ("outlier_whistleblower.txt", 0): "whistleblower report ETH-2025-3391's submission details",
    ("outlier_whistleblower.txt", 1): "the allegation summary in whistleblower report ETH-2025-3391",
    ("outlier_whistleblower.txt", 2): "the investigation status of whistleblower report ETH-2025-3391",
    ("outlier_whistleblower.txt", 3): "the risk rating in whistleblower report ETH-2025-3391",
    ("outlier_travel.txt", 0): "the travel security incident details for Amara Okonjo",
    ("outlier_travel.txt", 1): "the resolution of Amara Okonjo's travel security incident",
}

HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &,.'\-]{2,60}$")


def fetch_all_points(client):
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    return points


def is_header_line(line):
    line = line.strip()
    return bool(line) and bool(HEADER_RE.match(line)) and not any(c.isdigit() for c in line[:3])


LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z()\s',.\-]*)")


def extract_line_label(line):
    """Pull the leading text label off a line like 'Net income $ 112,010 ...'."""
    m = LABEL_RE.match(line.strip())
    if not m:
        return None
    label = m.group(1).strip().rstrip(":")
    return label if len(label) > 2 else None


def derive_control_topics(points):
    """
    Carry the last-seen section header forward across continuation chunks,
    but a section (e.g. "CONSOLIDATED STATEMENTS OF OPERATIONS") can span
    several chunks — without disambiguation, every continuation chunk of
    that section would get the identical topic phrase despite covering
    different line items, making their generated queries indistinguishable
    even though they target different chunks. So continuation chunks get
    the specific line-item label from their own first line appended.
    """
    by_doc = defaultdict(list)
    for p in points:
        by_doc[p.payload["metadata"]["source_file"]].append(p)

    topics = {}
    for source_file, doc_points in by_doc.items():
        doc_points.sort(key=lambda p: p.payload["metadata"]["chunk_index"])
        current_header = None
        for p in doc_points:
            first_line = p.payload["page_content"].split("\n")[0]
            if is_header_line(first_line):
                current_header = first_line
                topics[p.id] = current_header
            else:
                label = extract_line_label(first_line)
                if current_header and label:
                    topics[p.id] = f"{current_header} (specifically {label.lower()})"
                else:
                    topics[p.id] = current_header
    return topics


FIGURE_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million))?(?:\s*per share)?")
BARE_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")


def extract_distinguishing_figure(page_content):
    """
    A dollar figure from the SAME line/paragraph the topic label itself
    came from -- not from anywhere in the chunk. The control template
    names the company, but every same-template company shares identical
    section headers ("DIVIDENDS", "FINANCIAL HIGHLIGHTS FY2023") -- a
    query built only from company + header matches the document's
    opening chunk (which prominently repeats the company name) just as
    strongly as it matches the actual target section, regardless of
    topic. Anchoring on a specific figure disambiguates -- but a chunk
    can contain more than one line item (e.g. "Gross margin ..." followed
    later, in the same chunk, by "Net income $112,010 ..."), and
    searching the whole chunk grabs whichever number happens to have a
    "$" first, which may belong to a completely different line item than
    the one actually labeled. Restricting the search window to match
    where derive_control_topics() drew the label from keeps the two
    consistent.
    """
    first_line = page_content.split("\n")[0]
    if is_header_line(first_line):
        # Header chunk -- the label is the header; restrict to this
        # section's own paragraph so a second section later in the same
        # chunk (e.g. "FINANCIAL HIGHLIGHTS FY2022" tacked onto the end)
        # can't supply the figure instead.
        window = page_content.split("\n\n")[0]
    else:
        # Continuation chunk -- the label came from this exact line.
        window = first_line

    m = FIGURE_RE.search(window)
    if m:
        return m.group(0)
    m = BARE_NUMBER_RE.search(window)
    return f"${m.group(0)}" if m else None


def build_query(chunk, category, control_topics):
    metadata = chunk.payload["metadata"]
    source_file = metadata["source_file"]

    if category == "outlier":
        topic = OUTLIER_CHUNK_TOPICS[(source_file, metadata["chunk_index"])]
        return f"What information does the document contain about {topic}?"

    company = COMPANY_NAMES[source_file]
    header = control_topics.get(chunk.id) or "financial results"
    figure = extract_distinguishing_figure(chunk.payload["page_content"])
    topic = f"{company}'s {header.lower()}"
    if figure:
        topic += f", which reports a figure of {figure}"

    return f"What information does the document contain about {topic}?"


def stratified_control_sample(control_points, n):
    by_doc = defaultdict(list)
    for p in control_points:
        by_doc[p.payload["metadata"]["source_file"]].append(p)
    for doc_points in by_doc.values():
        doc_points.sort(key=lambda p: -p.payload["metadata"].get("corpus_frequency", 0))

    docs = list(by_doc.keys())
    sample = []
    i = 0
    while len(sample) < n and any(by_doc.values()):
        doc = docs[i % len(docs)]
        if by_doc[doc]:
            sample.append(by_doc[doc].pop(0))
        i += 1
        if i > n * len(docs) * 2:
            break
    return sample[:n]


def main():
    client = QdrantClient(host="localhost", port=6333)
    points = fetch_all_points(client)

    # Restrict "outlier" to the purpose-built singleton documents only.
    # A few chunks in control-genre docs (e.g. apple, sableridge) also land
    # in the `outlier` category incidentally, because their specific wording
    # happened to have few neighbours -- that's not the same as being a
    # genuinely rare/sensitive record, and mixing them in would confound
    # "rare document" with "generic prose that happens to score low".
    outlier_points = [
        p for p in points
        if p.payload["metadata"].get("category") == "outlier"
        and (p.payload["metadata"]["source_file"], p.payload["metadata"]["chunk_index"]) in OUTLIER_CHUNK_TOPICS
    ]
    control_points = [p for p in points if p.payload["metadata"].get("category") == "control"]

    control_topics = derive_control_topics(points)
    control_sample = stratified_control_sample(control_points, CONTROL_SAMPLE_SIZE)

    dataset = []
    qid = 1
    for category, chunks in (("outlier", outlier_points), ("control", control_sample)):
        for chunk in chunks:
            metadata = chunk.payload["metadata"]
            dataset.append({
                "id": f"Q{qid:03d}",
                "category": category,
                "source_file": metadata["source_file"],
                "target_chunk_id": chunk.id,
                "target_chunk_index": metadata["chunk_index"],
                "corpus_frequency": metadata.get("corpus_frequency"),
                "chunk_preview": chunk.payload["page_content"][:120],
                "question": build_query(chunk, category, control_topics),
            })
            qid += 1

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {len(dataset)} queries to {OUTPUT_PATH}")
    print(f"  outlier: {len(outlier_points)}")
    print(f"  control: {len(control_sample)} (sampled from {len(control_points)})")


if __name__ == "__main__":
    main()
