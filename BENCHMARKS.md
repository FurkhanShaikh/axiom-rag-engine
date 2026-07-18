# Axiom Engine Benchmarks

Axiom's claim is that it verifies citations. That claim is only credible with
numbers, so this page is where the verification quality is measured and
published. The measurements come from the eval harness in [`evals/`](evals/),
which runs the **production** verification code path — not a reimplementation of
it — so the numbers reflect what a caller actually gets.

## What is measured

| Layer | What it answers | Dataset | Needs LLM keys |
|---|---|---|---|
| Retrieval quality | Does the ranker surface the right sources near the top? | [SciFact](https://github.com/allenai/scifact) dev split | No |
| Semantic verifier accuracy | Does the verifier pass faithful claims and fail unfaithful ones? | SciFact dev split | Yes |
| End-to-end golden set | Does the full pipeline answer, tier, and gate as specified? | `evals/golden/seed.jsonl` (16 diagnostic cases) | Deterministic subset: no |

### Retrieval metrics

Verification can only bless what retrieval finds, so ranking quality is measured
first. SciFact labels each claim with the corpus document(s) that answer it, so
the ranker is scored as a standard IR system:

| Metric | Meaning |
|---|---|
| `recall@k` | Fraction of gold documents found in the top k |
| `nDCG@10` | Ranking quality with position discounting (higher = gold docs nearer the top) |
| `MRR` | Mean reciprocal rank of the first gold document |

This runs with no keys and is fully deterministic, so it gates every PR — and
it is the baseline that hybrid retrieval and a reranker (roadmap Phase 1) must
beat.

### Semantic verifier metrics

The positive class is **"unfaithful detected"** — a CONTRADICT claim the
verifier correctly fails. This is the metric that matters: a verification
product that misses misrepresentations is worse than useless because it
launders them as verified.

| Metric | Meaning | Why it matters |
|---|---|---|
| `unfaithful_recall` | Fraction of unfaithful claims caught | Low recall = misrepresentations slip through as Tier 4 misses |
| `unfaithful_precision` | Fraction of flagged claims that were truly unfaithful | Low precision = faithful claims bounced into rewrite loops, wasting budget |
| `unfaithful_f1` | Harmonic mean of the two | Single headline number |
| `accuracy` | Overall agreement with SciFact labels | Coarse sanity check |
| `error_rate` | Examples where the verifier produced no verdict | Infra/parse reliability |

## Results

### Retrieval — SciFact dev

| Method | claims | recall@1 | recall@5 | recall@10 | recall@20 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|
| **BM25** (production ranker) | 188 | **0.656** | **0.870** | **0.916** | **0.948** | **0.797** | **0.763** |
| hybrid (BM25 + dense, RRF) | 188 | 0.587 | 0.842 | 0.904 | 0.944 | 0.758 | 0.719 |
| dense (`nomic-embed-text`) | 188 | 0.498 | 0.714 | 0.774 | 0.824 | 0.645 | 0.615 |

**Finding: hybrid does not beat BM25 on SciFact.** BM25 wins every metric;
adding dense retrieval via reciprocal-rank fusion makes it slightly *worse*
because the local embedder's signal is weaker and drags the fusion down.

This is a genuine result, not an implementation gap — the RRF math is unit-
tested, and dense does add *some* orthogonal value: it rescues 7 of 188 claims
BM25 misses in the top 10. But the complementary signal is thin. An oracle that
always picked the better ranker per query would reach recall@10 ≈ 0.968 vs
BM25's 0.931 — an upside of only **~3.7 points**, which real RRF can't capture
because it pays a larger penalty on the 35 claims where dense ranks the gold
document poorly.

Why: SciFact claims are lexically clean scientific text, exactly where BM25
excels and general-purpose embeddings add little. Hybrid retrieval's real
advantage is vocabulary mismatch (paraphrase, synonyms, colloquial vs formal),
which this dataset under-represents. **The recommendation is not to wire hybrid
into production on this evidence** — first get a benchmark that stresses
semantic matching and/or a stronger domain-appropriate embedder, then re-measure
with the same eval (`--method hybrid`). The measurement did its job: it stopped
a feature that doesn't help from shipping.

Only the **BM25** row is an enforced per-PR gate (deterministic, no keys); a
ranker change that drops any metric fails CI. Dense/hybrid are research runs
(they need a local embedder) and are not gated.

#### When does hybrid win? A controlled vocabulary-mismatch A/B

The SciFact-native result ("hybrid loses") is not the whole story. Two follow-up
experiments isolate *when* dense retrieval helps.

**1. Diagnostic (real data, no manipulation).** Grouping the 188 claims by
retrieval outcome and measuring the lexical overlap between each claim and its
gold document:

| outcome | n | mean query→gold overlap |
|---|---|---|
| both BM25 & dense find it | 140 | 0.669 |
| BM25 only | 35 | 0.573 |
| **dense rescues (BM25 missed)** | 7 | **0.342** |
| neither | 6 | 0.257 |

Dense rescues exactly the low-overlap claims — half the lexical overlap of what
BM25 handles. Dense's value is vocabulary-mismatch robustness.

**2. Causal A/B (paraphrased queries).** Rewording 80 claims with synonyms lowers
their lexical overlap with the gold document (0.64 → 0.51, −20%) while keeping
the answer document the same — isolating vocabulary mismatch. Matched on the
same 80 claims:

| method | original (r@10 / nDCG@10 / MRR) | paraphrased (r@10 / nDCG@10 / MRR) |
|---|---|---|
| BM25 | 0.906 / 0.783 / 0.746 | 0.877 / 0.764 / 0.740 |
| dense | 0.811 / 0.690 / 0.662 | 0.833 / 0.713 / 0.681 |
| **hybrid** | 0.877 / 0.762 / 0.735 | **0.880 / 0.783 / 0.766** |

**The crossover:** on original queries hybrid trails BM25 (nDCG −2.0, MRR −1.1);
on paraphrased queries hybrid overtakes it (nDCG **+1.9**, MRR **+2.6**). BM25
degrades under paraphrase (recall@10 −2.9) while hybrid holds flat — dense
compensates when lexical matching falters.

**What this means for Axiom.** Production retrieval is web search: colloquial
user queries against formal pages — a *high* vocabulary-mismatch regime, exactly
where hybrid wins. So hybrid retrieval is worth pursuing for the real workload,
even though it loses on lexically-clean SciFact.

**Honest caveats.** Magnitudes are modest (n=80, a small local embedder
`nomic-embed-text`), so treat the numbers as directional, not precise. The
paraphrase manipulation also confounds two effects — it lowers lexical overlap
(hurts BM25) *and* makes the terse SciFact claims more fluent (helps the
embedder); the clean diagnostic above isolates the overlap mechanism on its own.
The recommendation is therefore **not** an immediate production commit but the
next measurement: build a web-search-flavored retrieval eval (real colloquial
queries) and/or try a stronger embedder, then re-run `--method hybrid`. The
machinery is in place; reproduce with `evals/make_paraphrases.py` +
`retrieval_eval.py --paraphrased`.

### Verification

> **Not yet recorded for the production model.** The semantic table is populated
> from the first keyed run on the production verifier. Until then the semantic
> gate ships `report_only` and does not block CI. See **Recording the baseline**
> below.

### Semantic verifier — SciFact dev

| Model | n | recall | precision | f1 | accuracy | error_rate | recorded |
|---|---|---|---|---|---|---|---|
| `gpt-4o-mini` (production) | — | — | — | — | — | — | _pending_ |
| `google/gemma-4-26b-a4b:free` (via OpenRouter) | 30 | 0.857 | 0.667 | 0.750 | 0.733 | 0.00 | 2026-07-17 |

> The gemma row is a **provisional reference point on a free model**, not the
> production verifier, on a small (n=30) sample. It exists to prove the harness
> produces real numbers end to end. Read it as "the pipeline works and a
> mid-size open model catches ~86% of misrepresentations while over-flagging
> ~33% of faithful claims" — not as a production SLA. The production
> `gpt-4o-mini` numbers are still pending a keyed run (see below).

### End-to-end golden set

| Mode | cases | pass rate | cost | recorded |
|---|---|---|---|---|
| deterministic (`--validate-only`) | 16 | 16/16 | $0.00 | 2026-07-17 |
| full (keyed) | 16 | — | — | _pending_ |

The deterministic pass rate is gated on every PR (`ci.yml` → `evals-gate`); it
runs the retriever, scorer, ranker, and pre-LLM answerability gate with no keys
and must stay at 16/16.

## Running the evals

The eval scripts load `.env` automatically, so any provider key there
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) is picked up
without exporting it to the shell.

```bash
# One-time: fetch SciFact (~3 MB)
python tasks.py evals download

# Deterministic gate — no keys, this is the per-PR gate
python tasks.py evals gate

# Keyed semantic verifier accuracy (production model)
python tasks.py evals semantic -- --model gpt-4o-mini --limit 200

# Free model via OpenRouter (shared endpoints throttle — run serially).
# The eval retries transient 429s with the provider's Retry-After.
AXIOM_MAX_CONCURRENT_LLM=1 python tasks.py evals semantic -- \
  --model "openrouter/google/gemma-4-26b-a4b-it:free" --limit 30

# Keyed end-to-end golden set
python tasks.py evals e2e -- --model gpt-4o-mini
```

> **Free-tier note.** OpenRouter's free model pool is shared and heavily
> throttled for zero-credit accounts — many models return HTTP 429
> (`is temporarily rate-limited upstream`) regardless of your personal quota,
> and availability shifts by time of day and provider. Run serially
> (`AXIOM_MAX_CONCURRENT_LLM=1`), keep samples small (the free daily cap is
> ~50 requests), and expect to pick whichever free model is answering. The
> gemma row above was recorded this way.

See [`evals/README.md`](evals/README.md) for the harness internals and
[`evals/gate.py`](evals/gate.py) for the gate contract.

## Recording the baseline

The regression gate needs defensible floors before it can block. To activate
the semantic gate:

1. **Run it** on the production verifier model with a meaningful sample:
   ```bash
   python tasks.py evals semantic -- --model gpt-4o-mini --limit 200
   ```
2. **Read the summary** the run prints (recall, precision, f1, accuracy,
   error_rate).
3. **Set the floors** in `evals/baselines/semantic-verifier.json` a few points
   below each observed value — the `tolerance` band absorbs run-to-run noise, so
   the floor is the "never regress past here" line, not the observed number
   itself. Set `error_rate`'s ceiling a few points above observed.
4. **Fill in this page** — the results table above and the `recorded_at` /
   `model` fields in the baseline.
5. **Flip `enforcement` to `"enforce"`** in the baseline. From then on, a
   regression past the tolerance band fails the nightly job.

Do the same for a keyed `e2e-golden` baseline (`pass_rate` floor, `cost_usd`
ceiling) once you want the full keyed run gated nightly.

## Honest caveats

- **Small local models score lower.** Ollama models will underperform cloud
  models on the semantic layer — compare like against like, never a local run
  against a cloud baseline.
- **BM25 relevance is 0 for non-Latin queries** (the tokenizer is ASCII-only),
  so the Arabic and CJK golden cases rank on quality score alone. This is a
  known retrieval limitation, tracked for the hybrid-retrieval work.
- **Tier calibration is not yet measured.** Whether Tier 1 correlates with
  actual answer correctness needs labeled answer correctness, which the current
  seed set is too small to provide. That arrives with a larger pinned-corpus
  dataset (ALCE/ASQA).
