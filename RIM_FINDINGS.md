# Retrieval-Induced Memorization: When the Measurement Manufactures the Effect

*Three separate times in this experiment, a plausible-looking, large effect turned out to be a property of how it was measured rather than of the pipeline being measured — a query-construction artifact (Section 3), and two distinct scoring artifacts uncovered together in Section 6 (a fact-list construction bug, and a retrieval-hit/exposure decoupling). Section 10 states what generalizes from each beyond this corpus.*

## 1. Research question

**Hypothesis**: in RAG systems, documents with unusually low corpus frequency — few semantically similar neighbours in the corpus — get reproduced verbatim in LLM outputs at higher rates than common documents, even under normal, non-adversarial queries. If true, this is a privacy leak that sits outside the usual differential-privacy boundary, because it happens at inference time through retrieved context rather than through anything the model learned during training.

Three sub-hypotheses:
- **H1** — outlier (rare) chunks are reproduced more verbatim than frequency-matched control (common) chunks.
- **H2** — the cross-encoder reranker amplifies this, by scoring outlier chunks higher and pulling them into the generation context more often.
- **H3** — model-layer differential privacy provides no bound on this leak, since it happens via retrieval, not training.

This report covers H1 and H2 empirically. H3 is addressed as a discussion point (see Section 7) — it isn't independently testable with a query experiment, since it's a claim about what DP guarantees *don't* cover, not a measurable property of this pipeline.

## 2. Corpus design

The experiment needed two document populations, distinguished by **corpus frequency** — how many other chunks in the corpus are semantically similar (cosine similarity ≥ 0.85) to a given chunk:

- **Control-candidate documents**: 9 financial annual reports sharing a common template (Business Overview, Financial Highlights, Balance Sheet, Segment Revenue, Dividends, etc.) — one real (Apple's 10-K, trimmed to a representative excerpt) and eight fictional companies written to the same structure (Acme, Meridian, Bexley, Calloway, Thornbury, Vantage, Sable Ridge, Ridgemont). Same genre, different companies — so "high frequency" reflects genuine cross-document similarity, not one document's internal repetition.
- **Outlier documents**: 10 short (2-4 chunk), single-record confidential documents spanning distinct domains — a medical case, a personal health record, a legal memo, an HR disciplinary record, an engineering incident postmortem, a customer support transcript, a government benefits case, a grant proposal, a whistleblower report, a travel security incident.

All documents are synthetic (fictional company names/numbers, or a trimmed public 10-K) — no real personal data.

### Corpus-frequency tagging

`rim_analysis.py` scrolls every chunk out of Qdrant, computes the full pairwise cosine-similarity matrix, counts each chunk's neighbours at similarity ≥ 0.85, and stores that count (`corpus_frequency`) plus a derived tag — `outlier` (<3 neighbours), `control` (≥10 neighbours), or `unclassified` (in between) — back into each chunk's Qdrant payload. Final corpus:

| Arm | Documents | Chunks |
|---|---|---|
| Control-candidate | 9 (acme, apple, bexley, calloway, meridian, ridgemont, sableridge, thornbury, vantage) | 114 |
| Outlier | 10 (medical, personal, legal, HR, incident, support, benefits, grant, whistleblower, travel) | 25 |
| **Total** | **19** | **139** |

Applying the corpus-frequency tags across all 139 chunks gives **30 tagged `outlier`, 49 `control`, 60 `unclassified`** — the `outlier` tag count (30) is slightly higher than the 25 outlier-document chunks above, because a handful of chunks *inside* control-candidate documents (4 in `apple`, 1 in `sableridge`) also score as `outlier` incidentally, just from having unusually-worded phrasing — not because they're genuinely rare records. Section 3 excludes those 5 from the experiment's outlier arm for exactly that reason.

Two validation checks were run before trusting this design:
1. **Cross-document semantic matching, verified manually**: the `control` chunks that crossed the 0.85 threshold were spot-checked and are genuinely matched content — e.g., acme's and apple's "dividends" sections (similarity 0.895), not coincidental artifacts.
2. **Outlier mutual isolation**: full pairwise similarity across all 10 outlier documents confirmed none of them cluster with each other — max similarity anywhere in that matrix was 0.818, comfortably below the 0.85 threshold, so each is a genuinely independent low-frequency data point.

A corpus-viability checklist (≥8-10 independent documents per arm, ≥4-5 distinct outlier domains, ≥2-3 independent control sources, ≥150 total chunks) was applied before proceeding — the corpus met every bar except total volume, which landed at 139 against a 150 floor (close enough to proceed).

**Note**: `ds_salaries.csv` (a tabular salary dataset) was ingested during early exploration, found to fit neither category (all chunks landed `unclassified`), and was deliberately dropped from the final corpus rather than forced into either arm.

## 3. Query construction

55 queries were built (`build_rim_queries.py`): all 25 chunks from the 10 outlier documents that pass the purpose-built-singleton filter (excluding the 5 incidentally-tagged `apple`/`sableridge` chunks described above), plus a stratified sample of 30 `control` chunks (round-robin across the 9 control documents, prioritized by corpus frequency).

Each chunk gets one standardized, generic query — `"What information does the document contain about {topic}?"` — rather than a hand-written question per chunk, so the LLM decides how much to reproduce rather than being led toward one specific fact. Outlier topics are hand-mapped per chunk (each covers a distinct sub-topic within its document, e.g. diagnosis vs. follow-up care). Control topics are derived from each chunk's own section header.

### A methodological confound, found and fixed

The first version of this query set anchored control-document topics on **company name + section header** (e.g. *"...about Acme Financial Corporation's dividends?"*). This produced a control retrieval hit rate of only **16.7%** (reranked) / **10%** (no-rerank) — dramatically lower than outlier's 100%/60%.

Manual inspection of the failed queries (prompted by a sanity check before trusting these numbers) showed the cause: every same-template company's opening chunk contains the company's name as its literal title line. Since the query also contains the company name, that chunk wins retrieval almost regardless of which section was actually asked about — a **query-construction artifact**, not a corpus-frequency effect.

**Fix**: each control query now also includes a specific figure pulled from the *target chunk's own body text* (e.g. *"...about Acme Financial Corporation's dividends, which reports a figure of $4.20 per share?"*) — the same principle already used successfully for outlier queries (per-chunk, not per-document, topic phrases). This is a chunk-unique anchor that the company's title chunk doesn't share.

**Result of the fix**: control retrieval hit rate rose from 16.7% to **96.7%** (reranked) with no other change — confirming the original gap was almost entirely the query artifact, not a genuine retrieval-difficulty difference between outlier and control content.

**A second-order bug, also found and fixed**: the figure-extraction logic initially searched the *whole* chunk for the first dollar amount, rather than the specific line the topic label came from. For chunks spanning multiple line items, this occasionally anchored the query to the wrong figure (one query about "gross margin" ended up anchored to the adjacent "net income" figure instead). Fixed by restricting the search to the same line/paragraph the label itself was drawn from, and every one of the 55 final queries was audited to confirm its anchor figure actually appears in the labeled section.

## 4. Experiment design

`rim_experiment.py` runs each of the 55 queries through `RetrievalChain`'s internals directly (not the `/query` HTTP endpoint — see Section 8 for why), in two pipeline variants:

- **`reranked`** — production default: hybrid retrieval → cross-encoder rerank → top 3 to the LLM.
- **`no_rerank`** — hybrid retrieval → top 3 taken directly, reranking skipped.

For each variant, two things are recorded:
- **Retrieval hit** — did the target chunk survive into the actual generation context (tests H2).
- **Reproduction ratio** — the fraction of the target chunk's own 4-grams (word n-grams) that appear verbatim in the generated answer (tests H1). A continuous score, not binary, so it distinguishes "cited one figure" from "reproduced most of the source text."

All 55 queries completed with zero errors in the final run (paced at ~4.5s between generation calls to stay under the free tier's 15 requests/minute limit, with automatic resume-on-crash). Every query's `(source_file, target_chunk_index, target_chunk_id)` was cross-validated against `rim_queries.json` before analysis — all 55 targets are unique, and every stored result matches its query's intended target exactly.

## 5. Results — verbatim overlap (4-gram) metric

### H2 — does the reranker amplify outlier retrieval more than control?

| | Candidate pool | Reranked (production) | No-rerank |
|---|---|---|---|
| Outlier hit rate | 25/25 (100%) | 25/25 (100%) | 15/25 (60%) |
| Control hit rate | 30/30 (100%) | 29/30 (96.7%) | 21/30 (70%) |
| **Hit-rate lift from reranking** | — | **+40.0 points** | **+26.7 points** |

Fisher's exact test on final reranked hit rates: **p = 1.0** — no significant difference between categories.

**H2 is not supported.** The reranker improves retrieval broadly — both categories gain a large boost from reranking (+40pp outlier, +26.7pp control) — rather than preferentially amplifying outlier content specifically.

### H1 — are outlier chunks reproduced more verbatim than control chunks?

**Unconditional** (averaged over all 55 queries, regardless of whether the target chunk was actually retrieved):

| | Outlier | Control |
|---|---|---|
| Reranked | 0.226 | 0.203 |
| No-rerank | 0.147 | 0.083 |

**Conditional** (only among queries where the target chunk was actually retrieved — isolating reproduction *given* exposure, from retrieval-frequency effects):

| | Outlier (n) | Control (n) | Mann-Whitney U (one-sided, outlier > control) |
|---|---|---|---|
| Reranked | 0.226 (n=25) | 0.210 (n=29) | **p = 0.150** (not significant) |
| No-rerank | 0.243 (n=15) | 0.118 (n=21) | **p = 0.020** (significant at α=0.05) |

**H1 is supported only conditionally.** In the production configuration (reranker on), the gap between outlier and control reproduction is small and not statistically significant. Without the reranker, a real and significant gap emerges — outlier content that gets retrieved is reproduced roughly twice as faithfully as control content that gets retrieved.

### Continuous check: regressing reproduction on corpus_frequency directly

The outlier/control split is a binary simplification of a value that's actually continuous and already stored per chunk (`corpus_frequency`, ranging 0-2 for outliers and 12-32 for controls in this corpus). Regressing reproduction ratio on that raw value directly, across all 55 chunks combined, tests the same relationship with more statistical power than a two-group comparison, and distinguishes a graded relationship from a purely thresholded one:

| | n | OLS slope | r² | p (OLS) | Spearman ρ | p (Spearman) |
|---|---|---|---|---|---|---|
| Reranked, conditional on hit | 54 | +0.00001 | 0.000 | 0.998 | -0.043 | 0.758 |
| No-rerank, conditional on hit | 36 | -0.00632 | 0.123 | **0.036** | -0.253 | 0.136 |

This corroborates the categorical result from a different angle: in production (reranked) mode, reproduction ratio has essentially zero relationship with corpus frequency — the OLS slope is indistinguishable from flat. Without reranking, there's a real graded negative relationship (higher frequency → lower reproduction), significant by OLS though the rank-based Spearman test doesn't reach significance at this sample size — consistent with a real but noisy effect rather than a strict two-bucket threshold phenomenon.

## 6. Results — fact-level recall metric

### Why a second metric

4-gram word overlap has a structural bias that only became visible after inspecting individual answers: outlier chunks are prose (a diagnosis, a settlement figure, a case history), and paraphrasing that prose preserves its meaning while breaking most exact 4-gram matches. Control chunks are already-terse financial line items ("Total Revenue $4.82 billion"), which survive rewording far more easily because there's less structure to paraphrase in the first place. That means the metric under-counts disclosure for exactly the document type — dense, sensitive prose — that the RIM hypothesis is actually about.

Two examples make this concrete, and both turn out to be more informative than they first appear. `outlier_personal.txt` chunk 0 (frequency 0) — a personal health record containing menstrual cycle history, two prior pregnancies with dates, an IUD insertion date, a Pap smear result, BRCA test status, and a psychiatric medication with dosage — scored only **0.015** on the 4-gram metric, even though the model's answer restated essentially all of those facts in its own words (paraphrase breaking exact overlap, as expected).

The second example is stranger, and points at a different problem entirely. Asked about Apple's gross margin (chunk 8, control), the reranked pipeline answered completely and correctly — total gross margin, the products/services breakdown, the percentage — and scored **0.0** on 4-gram overlap. But `retrieved: false` for that variant: the target chunk was never in the final context. The correct figures came from a *different* chunk elsewhere in the document that happens to restate the same gross-margin fact in different words. This means retrieval-hit and reproduction are not the same thing: content can be fully disclosed via a semantically-overlapping neighbor even when the specific target chunk was never selected. Every "conditional on hit" comparison in Section 5 and below inherits this gap — "hit" tells you the target chunk specifically survived retrieval, not whether its content reached the answer through some other route. This doesn't invalidate those comparisons (the target chunk is still the most direct path for its content to appear), but it means the true "was this information exposed" rate is a floor, not an exact count, in both directions.

### Fact-level recall: method, and a construction bug that mattered as much as the metric choice

For each of the 55 target chunks, a fact list was hand-authored from the source content — 3-4 discrete, checkable items (specific IDs, names, dates, dollar figures) — scored as (facts recalled) / (facts in the chunk), where a fact counts as recalled if its keywords appear anywhere in the answer (case-insensitive substring match, not exact phrase).

The first version of these fact lists scored **0.840 outlier vs. 0.503 control (p < 0.0001)** in reranked mode — a dramatic, highly significant result. It was wrong. 19 of the 30 control target chunks span **two unrelated line-item sections crammed together by the text splitter** (financial highlights + balance sheet; dividends + long-term debt; Apple's gross margin + operating expenses + net income), but each query is anchored to only *one* of those sections. The first fact list included facts from both sections regardless — so a complete, correct answer about the one section actually asked about was scored as a *partial failure* for not also volunteering an unrelated line item that happened to share a chunk boundary. This is exactly the same shape of problem as the gross-margin example above (the Apple chunk-8 fact list originally included the net-income and R&D figures from later in the same chunk, none of which relate to "gross margin"), just discovered by auditing rather than by inspecting one striking answer.

**Fix**: each dual-topic chunk's fact list was rescoped to only the section its own query is anchored to (e.g. Acme chunk 3's facts are now its FY2021 revenue/net income/margin, not also the balance sheet cash/assets/current-ratio figures that follow in the same chunk).

### Results (corrected)

| | Outlier | Control | Mann-Whitney U (outlier > control) |
|---|---|---|---|
| Reranked | 0.840 (n=25) | 0.717 (n=30) | p = 0.170 (not significant) |
| No-rerank | 0.463 (n=25) | 0.594 (n=30) | p = 0.921 (not significant; direction reversed) |

The dramatic gap did not survive the fix. In production (reranked) mode, fact-level disclosure is *not* significantly different between outlier and control — consistent with the 4-gram metric's own (also non-significant) reranked result, rather than contradicting it as the first version of this analysis claimed. In no-rerank mode, fact-recall shows control numerically *higher* than outlier, also not significant — the opposite direction from the 4-gram metric's significant no-rerank result (p=0.020, Section 5).

That residual disagreement, specifically in no-rerank mode, is real and worth sitting with rather than explaining away: the 4-gram metric says outlier content that gets through keeps more of its *original phrasing* than control content does; fact-recall says outlier and control are disclosed with similar *factual completeness* regardless. Both can be true at once — a paraphrase can be just as complete as a verbatim copy, or an exact quote can convey only a fragment of the source. Phrasing-fidelity and information-completeness are dissociable, and this experiment's cleanest signal is that dissociation itself, not a clean verdict on H1 in either direction.

A residual, smaller-scale version of the query-framing asymmetry noted in earlier drafts of this analysis still exists — outlier queries ask broadly, control queries are figure-anchored (needed for retrieval reliability, Section 3) — but with fact lists now correctly scoped to what's actually asked, its effect on the numbers above is far smaller than the construction bug was, and doesn't change either result's significance.

## 7. Interpretation

Four independent analyses now sit together, and in the condition that matters most, they agree:

- **H2 is not supported** by any measure: the reranker improves retrieval broadly (Section 5), with no significant category-specific hit-rate gap.
- **H1, in production (reranked) mode, is not supported** by the 4-gram metric (p=0.150), the continuous regression (p=0.998, flat), *or* the corrected fact-recall metric (p=0.170). Three different ways of measuring reproduction, in the configuration that actually ships, all land on the same non-significant answer.
- **H1, with reranking off, gives conflicting signals depending on what's measured**: the 4-gram metric and the regression both show a real, significant effect (outlier reproduced more verbatim, p=0.020 and p=0.036) — but fact-recall shows no significant difference, numerically reversed. So even the one significant result in this whole experiment is specifically about *phrasing survival*, not about *whether the information got out* — and it only shows up in a configuration this pipeline doesn't actually run in production.

Two structural discoveries mattered more than any single number: retrieval-hit and information-exposure can decouple (content leaks via a semantically-redundant neighbor without its own chunk being selected, Section 6), and a chunk-boundary artifact in fact-list construction inflated an apparent p<0.0001 effect down to a non-significant one once corrected — twice in this project, first with queries (Section 3) and now with scoring (Section 6), the largest-looking effects turned out to be measurement artifacts, not signal. Taken at face value, **this corpus and this experiment do not support H1 in the configuration that matters** (production, reranked) under any of the three metrics tried. The one place a real effect survives (no-rerank, 4-gram/regression) is specific to verbatim phrasing, doesn't replicate in a paraphrase-robust metric, and describes a configuration not used in practice.

**H3** (model-layer DP doesn't bound this) is not independently tested here, but the mechanism observed in H1/H2 supports the underlying argument: whatever leak exists is happening through retrieval-time context assembly — which chunks the hybrid retriever and reranker select — entirely external to anything a model-weight DP guarantee could bound, since that leak exists regardless of what the generation model itself memorized during training.

## 8. A separate finding, outside RIM's scope

While building the direct-Python harness for this experiment, we found that the shipped `/query` endpoint's `sources` field is computed from a **separate, plain semantic search** (`similarity_search_with_relevance_scores`), not from the reranked context actually used to generate the answer. This means the "sources" shown to API users don't reliably reflect what the LLM actually saw — a pre-existing correctness issue in the production app, independent of this research, worth fixing separately.

## 9. Limitations

- **Sample size.** 25 outlier / 29-30 control chunks (after retrieval hits) is enough to detect a large effect (as the no-rerank result shows) but not a small one — any single reranked-mode "not significant" result, taken alone, could be a true null or simply underpowered. That ambiguity is weaker once three independently-computed measures (4-gram, continuous regression, fact-recall) all land on the same null in that condition (Section 7) — underpowering would need to suppress a real effect similarly across three metrics with different sensitivities, which is a less likely coincidence than any one of them individually missing a real but small effect. Convergence narrows the ambiguity; it doesn't eliminate it.
- **Synthetic corpus.** All outlier documents were authored for this study; no real personal data or real memorized training content was involved. This tests retrieval-time reproduction mechanics, not real-world memorization rates.
- **Single embedding/generation model.** Results are specific to `gemini-embedding-001` (retrieval) and `gemini-3.1-flash-lite-preview` (generation) — not necessarily generalizable to other model families.
- **Corpus-frequency metric.** The 0.85 cosine-similarity threshold and the <3/≥10 neighbour-count bands were chosen pragmatically (validated against the actual similarity distribution) rather than derived from a principled statistical rule.
- **Fact-recall keyword matching.** Substring-based fact matching is more paraphrase-robust than 4-gram overlap but still mechanical — it can be fooled by a keyword appearing without the fact it represents (e.g. a date mentioned in an unrelated sentence). A more serious version of this same failure mode — facts drawn from the wrong section of a multi-topic chunk — was caught and fixed in Section 6, but the fix relies on manual judgment about which section a query "really" targets, which is itself not perfectly objective.
- **Retrieval-hit is a floor, not an exact measure, of information exposure** (Section 6) — content can reach the answer via a semantically-redundant chunk without the specific target chunk being selected, so every "conditional on hit" comparison in this report (Section 5, Section 6) may undercount true exposure in both arms.
- **Query framing still differs structurally between arms** (Section 6) — control queries are figure-anchored (needed for retrieval reliability, Section 3), outlier queries are topic-broad. This residual asymmetry is much smaller than the construction bug that was fixed in Section 6, but wasn't eliminated by that fix.
- Two point IDs in `rim_queries.json`/`rim_results.json` were briefly corrupted twice during iterative fixes this session (a chunk-deduplication accident, then a result-migration key collision on the original duplicate pair) — both were caught by cross-validating every result against its query's intended target before trusting any number in this report, and both are fully resolved in the final dataset (0 inconsistencies across all 55 entries, verified directly). See `HOW_TO_RUN.md`'s troubleshooting section for the underlying duplicate-ingestion trap if reproducing from scratch.

## 10. What transfers beyond this corpus

The specific p-values here are tied to this corpus and won't generalize. Three things about *how those numbers went wrong* will, to any RAG evaluation built the same way:

- **A lexical anchor in a query can manufacture a retrieval effect that has nothing to do with what's being tested.** Section 3's confound wasn't corpus frequency — it was that every query for a templated document contained that document's own title text, which also happened to be the most prominent string in an unrelated "distractor" chunk (the cover page). Any evaluation that builds queries from document metadata (names, IDs, titles) should check whether that same text also appears, undiluted, somewhere it shouldn't — otherwise the eval is partly measuring "does this string appear verbatim in the corpus," not the property under test.
- **Exact-overlap metrics (n-gram, string-match) measure phrasing survival, not disclosure, and that gap is not random — it tracks source formatting.** Tabular, templated source text scores artificially high (numbers survive rewording as intact tokens) regardless of whether the answer actually contains more information. Paraphrased prose scores artificially low for the opposite reason. Whenever the groups being compared differ systematically in *how* their source content is structured — not just *what* it says — an exact-overlap metric is comparing formatting robustness, not the thing it's named after.
- **"Was the specific target chunk retrieved" is not a safe variable to condition on whenever the corpus contains redundant or overlapping content.** Section 6's Apple example shows the same fact stated in two different chunks; the model answered from the one that wasn't flagged as retrieved. Any evaluation that scores per-chunk retrieval hits, in a corpus where the same information can appear more than once, will undercount exposure — and if redundancy differs between the groups being compared, that undercount won't be symmetric.

## 11. Reproducing this

See `HOW_TO_RUN.md` for the full step-by-step guide. In short: `rim_analysis.py` → `build_rim_queries.py` → `rim_experiment.py` → `rim_regression.py` and `rim_fact_recall.py`, then read `rim_results.json` / `rim_fact_recall_results.json`.
