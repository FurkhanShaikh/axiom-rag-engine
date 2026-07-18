# Axiom Engine Evals

Measures whether the verification pipeline actually does what it claims —
as opposed to the unit suite, which measures whether the code does what the
code says. Run these before and after any change to prompts, models,
normalization, ranking, or tier logic.

## Layers

| Layer | Script | Dataset | What it measures |
|---|---|---|---|
| 1 | `semantic_verifier_eval.py` | [SciFact](https://github.com/allenai/scifact) (dev split) | Semantic verifier accuracy: does it pass SUPPORTed claims and fail CONTRADICTed ones? |
| 2 | `e2e_eval.py` | `golden/seed.jsonl` (committed, hand-curated) | Full pipeline behavior: answerability, tier assignment, Tier-5 leakage, non-Latin support |
| 3 | `retrieval_eval.py` | SciFact (dev split) | Retrieval quality: does the ranker surface the gold evidence document near the top? (recall@k, nDCG@10, MRR) |

### Why SciFact for Layer 3

SciFact's evidence labels name the corpus document(s) that answer each claim —
exactly the relevance signal an IR eval needs. Layer 3 ranks the whole corpus
per claim (reusing the production BM25 tokenizer, IDF, and constants) and scores
whether the gold documents land near the top. It is fully deterministic and
needs no keys, so it gates every PR. It is also the baseline hybrid retrieval
and a reranker must beat — see [BENCHMARKS.md](../BENCHMARKS.md). BM25 is already
strong on SciFact (lexically-clean scientific text), which makes it an honest,
demanding bar.

### Why SciFact for Layer 1

The semantic verifier's job — "does this claim faithfully represent this
quote in the context of this chunk?" — is exactly the claim-vs-evidence
entailment task. SciFact provides ~450 expert-annotated scientific claims
with labeled evidence sentences (SUPPORT / CONTRADICT) inside full
abstracts, which map directly onto (claim, exact_source_quote, chunk_text).
It is small, clean, and downloadable without extra dependencies.

Expansion candidates (not yet wired):
- **LLM-AggreFact** (HuggingFace) — aggregation of 10+ grounding datasets
  incl. RAGTruth and WiCE; much larger and more diverse than SciFact.
- **ALCE / ASQA** — questions with pinned retrieval passages; would let
  Layer 2 run on a standard corpus instead of only the golden seed.
- **CRAG** — web-flavored RAG QA with hallucination-aware scoring.

### Why a hand-curated golden set for Layer 2

No public benchmark exercises Axiom's specific contract (verbatim-quote
citations, six-tier rollup, unanswerable escape hatch, mock search
injection). The seed set is small but *diagnostic* — each case exists to
catch a specific regression class (see comments in `golden/seed.jsonl`).

## Running

```bash
# One-time: fetch SciFact (~3 MB) into evals/data/ (gitignored)
python tasks.py evals download

# Layer 1 — semantic verifier accuracy (needs an LLM; ~1-2k tokens/example)
python tasks.py evals semantic -- --model gpt-4o-mini --limit 50
python tasks.py evals semantic -- --model ollama/qwen3:8b --limit 20

# Layer 2 — end-to-end golden set (needs an LLM)
python tasks.py evals e2e -- --model gpt-4o-mini

# Layer 2 without any LLM: validates seed schema + runs the deterministic
# retriever/scorer/ranker stages and reports the pre-LLM answerability gate
python tasks.py evals e2e -- --validate-only

# Layer 3 — retrieval quality (ranks the SciFact corpus per claim)
python tasks.py evals retrieval                    # BM25, 100-claim sample (no LLM)
python tasks.py evals retrieval -- --limit 0       # BM25, all 188 labeled claims

# Dense / hybrid need a local Ollama embedder (default nomic-embed-text):
#   ollama pull nomic-embed-text
# The corpus is embedded once and cached to evals/data/embeddings/ (gitignored),
# so the first dense/hybrid run takes ~5 min and later runs are instant.
python tasks.py evals retrieval -- --method dense  --limit 0
python tasks.py evals retrieval -- --method hybrid --limit 0
```

**Retrieval methods.** `bm25` is the production ranker (deterministic, no keys,
the enforced gate). `dense` is cosine similarity over Ollama embeddings.
`hybrid` fuses BM25 and dense via reciprocal-rank fusion. On SciFact, BM25 wins
— see [BENCHMARKS.md](../BENCHMARKS.md) for the comparison and why. Dense/hybrid
are research runs, not gated.

Results are written to `evals/results/<eval>-<timestamp>.json` (gitignored)
with one record per example, so failures can be inspected and diffed
between runs.

## Metrics

**Layer 1 (semantic verifier)** — positive class = "unfaithful detected"
(CONTRADICT → semantic_check=failed):

- `accuracy` — overall agreement with SciFact labels
- `precision` / `recall` / `f1` — for the unfaithful class. Low recall means
  misrepresentations slip through (Tier 4 misses); low precision means
  faithful claims get bounced into rewrite loops (wasted LLM budget).
- `error_rate` — examples where the verifier LLM failed to produce a verdict

**Layer 2 (end-to-end)** — per-case expectation checks. A case's `expect`
block may assert any of:

- `answerable` — the `is_answerable` verdict must match
- `status_in` — response status must be one of the listed values
- `max_tier5` — Tier-5 leakage ceiling (usually 0)
- `max_tier1` — Tier-1 ceiling; used to prove a reference source (arXiv,
  Wikipedia) is *not* promoted to Authoritative
- `min_overall_score` — confidence floor for answerable cases

A case may also carry `app_config` and `pipeline_config` blocks, merged onto
the defaults, so a single harness can exercise trust policy (`banned_domains`,
`low_quality_domains`), `expertise_level`, and stage toggles without a bespoke
runner per class.

## The regression gate

`--gate` turns a run into a pass/fail CI decision against a committed baseline
in `evals/baselines/`. Each baseline lists the metrics that matter and a
per-metric bound (`floor` or `ceiling`) plus a `tolerance` slack band for
LLM run-to-run noise. See `evals/gate.py` for the full contract.

```bash
# Deterministic gates — no LLM keys, run on every PR. Two checks:
#   1. every golden case clears its pre-LLM answerability expectation
#   2. BM25 retrieval quality (recall@k / nDCG / MRR) does not regress
python tasks.py evals gate

# Keyed semantic gate — nightly. report_only until real floors are recorded.
python tasks.py evals semantic -- --model gpt-4o-mini --limit 200 --gate

# Retrieval gate on its own (also part of `evals gate`)
python tasks.py evals retrieval -- --limit 0 --gate
```

A baseline is `enforce` (a regression fails CI, exit 1) or `report_only`
(prints the comparison but never fails). Two baselines are `enforce` and need
no keys: `e2e-golden.json` (deterministic `validate_pass_rate`) and
`retrieval-bm25.json` (deterministic recall/nDCG/MRR). `semantic-verifier.json`
ships `report_only` with placeholder floors until the first keyed run records
defensible numbers; see `BENCHMARKS.md` for that one-time activation step.

The gate's exit code is the CI signal: 0 = pass, 1 = a regression on an
enforcing baseline (or a failing golden-case expectation).

## Interpreting results

Layer 1 (semantic) and the full Layer 2 (keyed) runs need LLM keys and cost
money, so they run nightly, not on every PR. The deterministic Layer 2 gate
(`--validate-only --gate`) runs on every PR. Treat the keyed numbers as a
baseline: record them before a change, re-run after, and investigate any
metric that moves more than the tolerance band. Small local models (Ollama)
will score lower on Layer 1 than cloud models — compare like against like.

## Known measurement gaps

- The ranker tokenizes `[a-z0-9]+` only, so BM25 relevance is 0 for
  non-Latin queries — such cases rank on quality score alone. The Arabic
  golden case exercises this path deliberately.
- Tier calibration (does Tier 1 correlate with actual correctness?) needs
  labeled answer correctness, which the seed set is too small to provide.
  That arrives with ALCE/ASQA integration.
