# Axiom Engine Benchmarks

Axiom's claim is that it verifies citations. That claim is only credible with
numbers, so this page is where the verification quality is measured and
published. The measurements come from the eval harness in [`evals/`](evals/).
The semantic-verifier, golden, and calibration evals run the **production** code
path. The retrieval eval does not yet: it re-implements BM25 over whole SciFact
abstracts (a fast path pinned to the production scoring function by a unit
test), and its hybrid and rerank methods are separate eval-side
implementations — so its numbers describe the ranking method, not the shipped
ranker's chunked, quality-blended ranking. Moving it onto the production ranker
is tracked in #51.

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
it is the baseline that hybrid retrieval and a reranker must beat.

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

**3. Confound-free confirmation (ArguAna, real queries).** The paraphrase A/B
uses synthetic rewrites. ArguAna (BEIR) is a real IR benchmark built for semantic
matching — each query is an argument and the gold document is its best
*counter*-argument, so query and answer are topically related but lexically
divergent. Running the same eval over its 8,674 docs and 1,406 real queries:

| method | recall@10 | recall@20 | nDCG@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.691 | 0.804 | 0.326 | 0.223 |
| dense (`nomic-embed-text`) | 0.599 | 0.745 | 0.277 | 0.193 |
| **hybrid** | **0.715** | **0.863** | **0.341** | **0.237** |

**Hybrid beats BM25 on every metric** (nDCG@10 **+1.5**, recall@20 **+5.9**, MRR
**+1.4**) on real, unmanipulated queries — the confound-free confirmation the
paraphrase A/B pointed to. Notably hybrid wins *even though dense alone loses* to
BM25 here: the local embedder is weak (its nDCG@10 of 0.277 is well below
nomic-embed-text's published ArguAna score of ~0.44 — Ollama's quantized build
underperforms the reference model), yet reciprocal-rank fusion still nets a gain
because dense surfaces gold documents BM25 misses. A production-grade embedder
would only widen the margin.

The BM25 number here (nDCG@10 0.326) matches the published BEIR leaderboard
(~0.31–0.40), a positive control that the eval itself is sound.

**What this means for Axiom.** Two datasets, opposite results, one rule: hybrid
loses on lexically-clean text (SciFact) and wins on vocabulary-mismatch text
(ArguAna, paraphrased SciFact). Production retrieval is web search — colloquial
user queries against formal pages, a *high* vocabulary-mismatch regime — so
hybrid retrieval is worth productionizing for the real workload. It clears the
bar even with a weak local embedder.

**Honest caveats.** The dense/hybrid runs use a weak local embedder
(`nomic-embed-text` via Ollama, a quantized build that underperforms the
reference model by ~35% on ArguAna), so the hybrid margins are a *floor*, not a
ceiling — a production embedder (OpenAI `text-embedding-3`, Cohere, Voyage, or a
properly-served open model) should do better. The paraphrase A/B additionally
confounds lower overlap with higher fluency; the ArguAna result above has neither
issue and is the load-bearing evidence. The SciFact dense/hybrid numbers predate
the embedder's task-prefix support; the prefix was measured neutral on Ollama's
build, so those numbers are essentially unchanged.

**Recommendation.** Hybrid retrieval is worth wiring into production for the
web-search workload — it wins on the vocabulary-mismatch data that matches that
regime, even with a weak embedder. The remaining prerequisite before shipping is
a production-grade embedder plus a re-run of these evals to size the real margin.
The machinery is in place: `retrieval_eval.py --method hybrid`, pluggable
embedder, and BEIR/paraphrase datasets to measure on.

#### Reranking — does a second-stage LLM re-order help?

A reranker re-scores the base ranker's top-K candidates with a model that sees
the query and passage *together* (a cross-encoder-style judgement), rather than
comparing independent vectors. `retrieval_eval.py --method rerank` grades each
(query, passage) pair 0–3 with a LiteLLM model and reorders by grade, keeping the
base order as a stable tiebreak — so it is a *refinement* of the base ranking,
never a reshuffle, and cannot change recall@k for k ≥ the rerank depth.

Matched 15-query paraphrased SciFact sample (same seed), reranking **BM25's
top-15** with `ollama/gemma4:e4b`:

| method | recall@1 | recall@5 | recall@10 | recall@20 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| BM25 (base) | 0.600 | 0.767 | 0.783 | 0.850 | 0.710 | 0.697 |
| hybrid | 0.683 | 0.850 | 0.850 | **0.933** | 0.792 | **0.795** |
| **rerank (BM25 base)** | **0.733** | 0.850 | 0.850 | 0.850 | **0.806** | 0.791 |

**Finding: reranking lifts precision substantially.** Over its BM25 base, rerank
gains nDCG@10 **+0.096** (0.710 → 0.806), recall@1 **+0.133** (0.600 → 0.733), and
MRR **+0.094** — pulling the right document to the *top*, exactly what a reranker
should do. It matches or slightly beats hybrid on every precision metric
(nDCG@10, recall@1) and ties on MRR.

**Structural limit: reranking cannot fix recall.** Rerank's recall@20 (0.850)
equals BM25's ceiling because it only reorders what BM25 retrieved — it can never
recover a gold document the base ranker missed entirely, which is why hybrid's
dense recall still wins recall@20 (0.933). The two techniques are complementary:
**hybrid widens the candidate net, reranking sharpens the top of it.** Reranking
*hybrid's* top-K (rather than BM25's) is the natural best-of-both, and the eval
supports it via `--rerank-base hybrid`.

**Cost reality (load-bearing for production).** This run made 225 grading calls
in **5,871 s — ~26 s per call** with a local *thinking* model (`gemma4:e4b` emits
a hidden reasoning trace before the digit; Ollama serializes requests, so the two
worker threads do not parallelize). That is fine for an offline eval but a
non-starter for a live API: reranking 15 candidates would add minutes of latency
per request. **The production reranker must be fast** — a hosted rerank/cross-
encoder API (Cohere, Voyage) or a small non-thinking model — not a local thinking
model. The quality signal is what generalizes here; the model choice does not.

**Caveats.** n = 15 is a small sample (chosen because each query costs ~6 min of
local grading), so treat the exact deltas as directional. The direction is robust
and matches the IR literature (second-stage reranking reliably improves
precision@k). Grades are disk-cached (`evals/data/rerank_cache/`, gitignored) so
re-runs and larger samples are incremental. Not gated: reranking needs an LLM.

**Recommendation.** Ship the reranker as an **opt-in** production stage
(`AXIOM_RERANKER_MODEL`, default off, fail-open to the base order), pointed at a
fast model the operator chooses — the same pattern as the synthesizer/verifier.
The eval justifies the feature; latency justifies keeping it off by default and
model-agnostic.

### Query expansion — live web (2026-09-22)

Do the retriever's extra searches earn their cost? `evals/query_expansion_eval.py`
ran the 15 answerable golden questions through the production retriever → scorer
→ ranker over **live Tavily results** (full-page content, BM25 ranking), in three
configurations. Web results carry no relevance labels, so the top-5 chunks were
graded 0–3 by an LLM judge (`ollama/qwen3.5:9b`, the reranker's grading prompt,
pooled across configurations).

| config | grade@5 | nDCG@5 | prec@5 (grade ≥ 2) | domains in top 10 | searches / question |
|---|---|---|---|---|---|
| original query only | 2.573 | **0.861** | 0.907 | 3.60 | **1.0** |
| legacy: + "What is q" + "Explain q" | 2.507 | 0.828 | 0.893 | **4.07** | 2.6 |
| + one LLM keyword rewrite | **2.613** | 0.856 | **0.933** | 3.80 | 2.0 (+1 LLM call) |

**Decision.** The legacy reformulations (production until this run) spent 2.6×
the searches and were slightly *worse* on every relevance metric; their only
gain was ~0.5 more distinct domains. They were removed: the first retrieval pass
now sends the original query only (~60% fewer web searches). The LLM rewrite
was a tie with original-only on relevance, so it was not worth an extra search
plus an LLM call. Retry-pass reformulations were kept — a retry skips URLs it
has already seen, so the original query alone would surface nothing new.

**Caveats.** 15 questions is a small sample and the judge is a local 9B model:
read this as "no benefit", not "proven harm". Live search results drift; the
Tavily responses behind these numbers are cached in
`evals/data/query_expansion_cache.json` (gitignored) so the run can be re-graded
with another judge at no search cost.

### Tier calibration — ASQA (2026-09-22)

`evals/calibration_eval.py` ran 100 ASQA dev questions (seed 0) through the full
pipeline: live Tavily retrieval (cached), `ollama/qwen3.5:9b` as synthesizer and
verifier. Every sentence was then judged against the full text of its cited
passages by a *different* model, `ollama/gemma4:e4b`. Two runs on the same
questions and the same retrieved sources: before and after the synthesizer
change that lets it see each chunk's source and asks it to cite every
supporting source (up to 3).

**Tier distribution**

| | before | after |
|---|---|---|
| sentences citing ≥ 2 domains | 0.6% | **23.5%** |
| Tier 1 / 2 / 3 share | 4.1% / 0% / 93.9% | 4.1% / **20.0%** / 71.8% |
| Tier 4 / Tier 5 share | 0.3% / 1.3% | 1.6% / 1.6% |
| citations per cited sentence | 1.03 | 1.40 |
| pipeline errors / unanswerable | 5 / 13 | 1 / 13 |

**Judged support per tier (after the change)**

| tier | n | judged supported | 95% CI | confidence weight |
|---|---|---|---|---|
| T1 Authoritative | 10 | 1.00 | [0.72, 1.00] | 1.00 |
| T2 Multi-Domain | 49 | 0.98 | [0.89, 1.00] | 0.85 |
| T3 Model Assisted | 176 | 0.95 | [0.91, 0.98] | 0.60 |
| T4 Misrepresented | 4 | 1.00 | [0.51, 1.00] | 0.20 |
| T5 Hallucinated | 4 | 0.75 | [0.30, 0.95] | 0.00 |

Before the change T3 measured 0.94 [0.91, 0.96] (n = 295) and T1 1.00 (n = 13).

**Answer level (non-circular).** STR-EM against ASQA's gold short answers was
0.385 → 0.352 for successful answers (answers got ~22% shorter, so they cover
fewer of ASQA's deliberately multiple interpretations; the difference is within
noise at n = 78). Spearman(overall_score, STR-EM) was −0.03 before and +0.13
after — **the confidence score does not predict whether an answer is right.**

**What this does and does not show.**

- *Tier 2 is now reachable* and its sentences are judged as well supported as
  Tier 3's, so the synthesizer change was kept.
- *Every verified tier is judged ~95–100% supported*, and the T1/T2/T3
  differences sit inside the confidence intervals. The judge also rated 3 of 4
  Tier 5 sentences "supported" — possible when a claim is right but its quote
  was not verbatim, but also a sign that this 4B local judge is lenient. With
  this judge the data **cannot distinguish the tiers, so the tier weights were
  not changed**: re-weighting on it would be tuning to noise. Re-judge the
  cached runs with a stronger model (`--phase judge --judge <model>`, no
  pipeline cost) before any weight change.
- *The confidence score measures grounding, not correctness or completeness.*
  A fully cited, faithful answer can still address the wrong interpretation of
  a question; nothing in the score captures that.

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
# Rate limits are retried by the production call path (AXIOM_LLM_MAX_RETRIES,
# honouring Retry-After); raising the retry settings here makes the run finish
# but its error rate then describes that policy, which the results record.
AXIOM_MAX_CONCURRENT_LLM=1 AXIOM_LLM_MAX_RETRIES=5 AXIOM_LLM_RETRY_MAX_WAIT_SECONDS=40 \
  python tasks.py evals semantic -- \
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
- **CJK lexical matching is approximate.** The BM25 tokenizer is Unicode-aware
  (Arabic, accented Latin, …) and splits CJK/Thai runs into character bigrams —
  the standard dictionary-free approach, but weaker than a real segmenter. It
  did not change the SciFact numbers above (English).
- **Tier calibration is only as good as its judge.** The ASQA calibration
  above used a local 4B judge that rated nearly everything supported; it cannot
  separate the tiers. A stronger judge is the next measurement, and the tier
  weights are unvalidated until then.
