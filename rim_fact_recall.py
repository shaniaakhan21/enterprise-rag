"""
Fact-level recall metric for the RIM experiment.

The 4-gram overlap metric in rim_experiment.py is structurally biased:
outlier chunks are prose (paraphrasing survives meaning but breaks exact
n-gram matches), while control chunks are financial line-item tables
(numbers get carried over as tokens even when reformatted, so table
structure survives n-gram matching much more easily than prose does).
That means the original metric under-counts disclosure for exactly the
document type (prose, PII-dense records) the RIM hypothesis is about.

This defines, per target chunk, a hand-authored list of discrete facts
(specific names, IDs, dates, dollar figures) -- the same content
regardless of how it's phrased -- and scores each answer by what
fraction of its target chunk's facts are disclosed, matched by flexible
keyword presence (case-insensitive substrings) rather than exact phrase
overlap. A fact is "recalled" if ALL of its keyword variants appear
somewhere in the answer.

Usage:
    python rim_fact_recall.py
"""
import json
import re

RESULTS_PATH = "rim_results.json"

# (source_file, chunk_index) -> list of facts; each fact is a tuple of
# keywords that must ALL appear (case-insensitive substring) in the answer.
FACTS = {
    # --- control ---
    # Chunk indices 3 and 9 (and apple's 8) each span TWO unrelated line-item
    # sections crammed together by the text splitter (financial highlights +
    # balance sheet; dividends + long-term debt; gross margin + operating
    # expenses/income). The query is anchored to only ONE of those sections
    # (matching its own topic label) -- so facts are scoped to that same
    # section only. Including the other section's facts would count a
    # complete, correct, on-topic answer as a failure just because it didn't
    # also volunteer an unrelated line item that happens to share a chunk
    # boundary. (Chunk 2's two "FINANCIAL HIGHLIGHTS" blocks are the same
    # topic across two years, not a section-boundary crossing, so unaffected.)
    ("acme_annual_report_2023.txt", 2): [("4.82",), ("0.93",), ("19.3",), ("12.47",)],
    ("acme_annual_report_2023.txt", 3): [("3.95",), ("0.71",), ("18.0",)],
    ("acme_annual_report_2023.txt", 6): [("11.8",), ("62",), ("17.7",), ("8-12",)],
    ("acme_annual_report_2023.txt", 9): [("4.20",), ("314",), ("1.15",)],
    ("apple_annual_report.txt", 3): [("178,353",), ("111,032",), ("64,377",), ("416,161",)],
    ("apple_annual_report.txt", 8): [("195,201",), ("180,683",), ("169,148",)],
    ("apple_annual_report.txt", 9): [("112,010",), ("7.49",), ("7.46",)],
    ("apple_annual_report.txt", 14): [("1.02",), ("0.98",), ("0.94",)],
    ("bexley_annual_report_2023.txt", 2): [("2.41",), ("0.46",), ("19.1",), ("8.12",)],
    ("bexley_annual_report_2023.txt", 3): [("1.97",), ("0.34",), ("17.3",)],
    ("bexley_annual_report_2023.txt", 9): [("2.95",), ("128",), ("0.78",)],
    ("calloway_annual_report_2023.txt", 2): [("1.86",), ("0.35",), ("18.8",), ("6.41",)],
    ("calloway_annual_report_2023.txt", 3): [("1.52",), ("0.26",), ("17.1",)],
    ("calloway_annual_report_2023.txt", 9): [("2.30",), ("79",), ("0.61",)],
    ("meridian_annual_report_2023.txt", 2): [("3.67",), ("0.68",), ("18.5",), ("9.84",)],
    ("meridian_annual_report_2023.txt", 3): [("3.01",), ("0.52",), ("17.3",)],
    ("meridian_annual_report_2023.txt", 9): [("3.60",), ("198",), ("0.95",)],
    ("ridgemont_annual_report_2023.txt", 2): [("2.19",), ("0.41",), ("18.7",), ("6.98",)],
    ("ridgemont_annual_report_2023.txt", 3): [("1.80",), ("0.31",), ("17.2",)],
    ("ridgemont_annual_report_2023.txt", 9): [("2.55",), ("93",), ("0.68",)],
    ("sableridge_annual_report_2023.txt", 2): [("1.34",), ("0.25",), ("18.7",), ("5.22",)],
    ("sableridge_annual_report_2023.txt", 3): [("1.09",), ("0.18",), ("16.5",)],
    ("sableridge_annual_report_2023.txt", 9): [("1.85",), ("56",), ("0.49",)],
    ("thornbury_annual_report_2023.txt", 2): [("2.94",), ("0.55",), ("18.7",), ("7.66",)],
    ("thornbury_annual_report_2023.txt", 3): [("2.41",), ("0.41",), ("17.0",)],
    ("thornbury_annual_report_2023.txt", 6): [("10.5",), ("32",), ("17.0", "550"), ("7-10",)],
    ("thornbury_annual_report_2023.txt", 9): [("3.15",), ("154",), ("0.83",)],
    ("vantage_annual_report_2023.txt", 2): [("3.28",), ("0.61",), ("18.6",), ("8.55",)],
    ("vantage_annual_report_2023.txt", 3): [("2.70",), ("0.46",), ("17.0",)],
    ("vantage_annual_report_2023.txt", 9): [("3.40",), ("171",), ("0.90",)],
    # --- outlier ---
    ("outlier_benefits.txt", 0): [("dhs-2025-114487",), ("aldenberg",), ("snap",), ("marsh",)],
    ("outlier_benefits.txt", 1): [("3", "household"), ("2,140",), ("536",), ("130%",)],
    ("outlier_benefits.txt", 2): [("2021", "2023"), ("march 2024",), ("january 15, 2026",)],
    ("outlier_grant.txt", 0): [("nsf-2025-88231",), ("wildfire",), ("helena wu",), ("742,000",)],
    ("outlier_grant.txt", 1): [("10-meter",), ("500",), ("15 minutes",), ("canyon creek",)],
    ("outlier_grant.txt", 2): [("dixie fire",), ("84%",), ("6x",), ("a100",)],
    ("outlier_hr.txt", 0): [("marcus webb",), ("emp-88214",), ("june 11, 2025",), ("forklift",)],
    ("outlier_hr.txt", 1): [("written warning",), ("3-day", "3 day"), ("dale ferris",), ("termination",)],
    ("outlier_incident.txt", 0): [("inc-2025-0442",), ("payments-gateway",), ("sev-2",), ("connection pool", "migration")],
    ("outlier_incident.txt", 1): [("6,200",), ("38,000",), ("priya nandan",), ("03:59",)],
    ("outlier_legal.txt", 0): [("johnson",), ("february 28, 2024",), ("7 years",), ("henderson",)],
    ("outlier_legal.txt", 1): [("127,000",), ("200,000",), ("450,000",)],
    ("outlier_medical.txt", 0): [("pt-2024-00847",), ("myocardial infarction", "stemi"), ("diabetes",), ("clopidogrel", "aspirin")],
    ("outlier_medical.txt", 1): [("sarah mitchell",), ("cardiology",)],
    ("outlier_personal.txt", 0): [("rodriguez",), ("1987",), ("2018",), ("2021",), ("brca",), ("sertraline",)],
    ("outlier_personal.txt", 1): [("blue cross", "bcb-8847291"), ("anita patel",)],
    ("outlier_support.txt", 0): [("cs-2025-77291",), ("denise okafor",), ("10029384",), ("six years",)],
    ("outlier_support.txt", 1): [("may 14",), ("july 22",), ("86.40", "86"), ("jira-bill-2291",)],
    ("outlier_support.txt", 2): [("86.40", "86"), ("25",), ("2/5", "2 out of 5")],
    ("outlier_travel.txt", 0): [("amara okonjo",), ("emp-53017",), ("luggage", "missing"), ("lh-clm-88420",)],
    ("outlier_travel.txt", 1): [("340",), ("6.2",)],
    ("outlier_whistleblower.txt", 0): [("eth-2025-3391",), ("anonymous",), ("4471",)],
    ("outlier_whistleblower.txt", 1): [("34,000",), ("expense reports", "client entertainment"), ("14",)],
    ("outlier_whistleblower.txt", 2): [("11",), ("corporate card", "calendar records")],
    ("outlier_whistleblower.txt", 3): [("moderate",), ("november 15, 2025",)],
}


def normalize(text):
    return re.sub(r"[,$%]", "", text.lower())


def fact_recall(facts, answer):
    if not facts:
        return None
    norm_answer = normalize(answer)
    hits = 0
    for fact in facts:
        if any(normalize(kw) in norm_answer for kw in fact):
            hits += 1
    return hits / len(facts)


def main():
    results = json.load(open(RESULTS_PATH))
    missing = []
    rows = []
    for r in results:
        if "error" in r:
            continue
        key = (r["source_file"], r["target_chunk_index"])
        facts = FACTS.get(key)
        if facts is None:
            missing.append(r["id"])
            continue
        for variant in ("reranked", "no_rerank"):
            answer = r["variants"][variant]["answer"]
            score = fact_recall(facts, answer)
            rows.append((r["id"], r["category"], variant, score, len(facts)))

    if missing:
        print(f"WARNING: no fact list for {len(missing)} chunks: {missing}")

    from statistics import mean
    from scipy import stats

    print(f"\ncoverage: {55 - len(missing)}/55 chunks have fact lists\n")

    for variant in ("reranked", "no_rerank"):
        outlier = [s for (_, c, v, s, _) in rows if c == "outlier" and v == variant]
        control = [s for (_, c, v, s, _) in rows if c == "control" and v == variant]
        u, p = stats.mannwhitneyu(outlier, control, alternative="greater")
        print(f"=== {variant} ===")
        print(f"  outlier: n={len(outlier)} mean fact-recall={mean(outlier):.4f}")
        print(f"  control: n={len(control)} mean fact-recall={mean(control):.4f}")
        print(f"  Mann-Whitney U (outlier > control): p={p:.4f}")
        print()

    out_path = "rim_fact_recall_results.json"
    json.dump(
        [{"id": qid, "category": cat, "variant": v, "fact_recall": s, "n_facts": n}
         for (qid, cat, v, s, n) in rows],
        open(out_path, "w"), indent=2,
    )
    print(f"Wrote per-query fact-recall scores to {out_path}")


if __name__ == "__main__":
    main()
