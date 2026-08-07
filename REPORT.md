# Technical report

How the system works, why it is built this way, and what it actually scores. Setup and
usage live in [README.md](README.md).

---

## 1. The data decides the design

Eight files. Read them and the shape of the problem is obvious:

| File | Content | Verdict |
|---|---|---|
| case_1 | Phishing → stolen wallet credentials | **Relevant** — initial access |
| case_2 | Solana wallet, funds split across addresses | **Relevant** — laundering |
| case_3 | Credential stuffing, 20 failed logins, Russian IP | **Relevant** — reconnaissance |
| case_4 | Keylogger malware — *"no direct link … to the stolen funds"* | **Distractor** |
| case_5 | Tornado Cash mixer, five intermediary wallets | **Relevant** — laundering |
| case_6 | Blocked SQL injection — *"no connection to the stolen … funds"* | **Distractor** |
| case_7 | Bitcoin ransom note after the attack | **Partial** — extortion, not theft |
| case_8 | Minor metadata breach two weeks earlier — *"no direct connection"* | **Distractor** |

Three of the eight distractors state their own irrelevance in plain English, and all eight
share the same security-and-finance vocabulary. So the hard problem here is **not finding
relevant text** — every file is about a cyber incident at a crypto exchange. It is
**refusing text that is topically perfect and evidentially worthless**.

That reframing drives every decision below: the system is built around a precision gate,
not around a better ranker.

---

## 2. Retrieval strategy: which, and why

The brief offers single-step or multi-step. **Both are implemented and switchable**, at the
CLI (`--mode`) and in the UI. The default is multi-step agentic.

### Single-step (`--mode single`)

One hybrid search on the question → rerank → threshold → shortlist → synthesise. One LLM
call, roughly a second. Correct default when the question already names what it wants, and
it is the honest baseline the agentic mode has to beat.

### Multi-step agentic (`--mode agentic`, default)

Not query expansion — a **closed loop with a stopping condition**:

```
gap ← "nothing collected yet"
for round in 1..max_rounds:
    query, why  ← PLANNER(question, queries tried, evidence so far, gap)
    candidates  ← hybrid_search(query) → rerank(against the question)
    shortlist   ← above threshold, unseen, ≤2 per source file
    verdict     ← GATEKEEPER(question, evidence so far, shortlist)
                    → admit[] with reasons
                    → reject[] with reasons
                    → sufficient? / what is still missing
    evidence    += admitted
    if verdict.sufficient or evidence full or (nothing admitted and round > 1): stop
    gap ← verdict.gap
```

Why a loop rather than expanding the query once and retrieving in parallel:

- **The stopping condition is the point.** Fan-out retrieval always returns *k* passages
  whether or not they deserve to exist. The loop can conclude "I have enough" — or "the
  archive does not answer this" — and both are real answers a detective needs.
- **The gatekeeper reasons over the shortlist, not one passage at a time.** It sees what
  is already collected, so it can reject a passage as redundant, not merely irrelevant.
- **Each round is steered by what the last one missed.** The planner is told the named gap,
  so round 2 is a targeted follow-up rather than a synonym of round 1.
- **It shows its work.** Every query, admission and rejection carries a reason, exposed in
  the UI's retrieval trace and in `--trace`. On an evidential task, "why is this here?"
  matters as much as the answer.

Rounds are capped (`max_rounds`, default 4) so a bad plan cannot loop forever, and reranking
is always scored against the **original question**, never the round's query — the query is a
way of finding material; relevance is only ever relevance to what was asked.

### The retrieval core, shared by both modes

**Dense** (`text-embedding-3-small`) generalises over paraphrase: a question about
"laundering" matches a passage that never uses the word. **BM25** catches literal tokens an
investigator types that embeddings blur — IP addresses, wallet names, `203.0.113.7`.
**Reciprocal rank fusion** combines them without having to calibrate a cosine against a BM25
score. **Cohere `rerank-v3.5`** is a cross-encoder: it reads query and passage jointly,
which a bi-encoder structurally cannot, and it is what makes a meaningful score threshold
possible at all.

One addition worth calling out: a **per-source cap** on the shortlist (default 2 passages
per file). Ranking alone is quietly biased towards whichever document repeats the query's
vocabulary most; on a small archive one file can fill the shortlist while a second file
holding the other half of the answer never reaches the gatekeeper. This was not theoretical
— before the cap, `case_2` took three of five shortlist slots on the laundering question and
**the Tornado Cash passage never surfaced at all**. With the cap it does.

### Storage: in-memory, deliberately

BM25 and the vector index are both **in-process** — a NumPy matrix and a ~50-line Okapi
BM25. The corpus is 8 files / 24 paragraphs, so brute-force cosine is *exact* and instant,
and a vector database would add operational weight, a network hop and a dependency in
exchange for nothing measurable.

This is a considered choice for the size of the data, not a limitation of the design. Both
sit behind narrow interfaces (`VectorIndex`, `BM25Index`), so the upgrade path is a
one-class change with no impact on the pipeline:

- **Vectors** → Qdrant, Chroma, pgvector or OpenSearch kNN once the corpus outgrows memory
  or needs to persist across processes.
- **Full text** → OpenSearch / Elasticsearch or Postgres FTS for real BM25 at scale, with
  analysers, stemming and phrase queries.
- **Fusion, reranking, the agent loop and citations are all storage-agnostic** and would not
  change.

Vectors are cached to `.cache/` keyed on corpus content **and embedder identity** (see §6).

---

## 3. Citation strategy

Citations follow OpenAI's
[citation-formatting convention](https://developers.openai.com/api/docs/guides/citation-formatting).
Evidence is handed to the synthesiser as numbered sentences:

```
[E1] source: case_5
L1: The stolen cryptocurrency was quickly moved through Tornado Cash…
L2: The exchange's security team noticed that within 30 minutes…
```

and the model writes prose with inline markers using private-use delimiters
(`citeE1L2`) that cannot collide with anything it would
legitimately write. We extract them with offsets, strip them from the text, and validate
each one against the evidence actually supplied.

**Why this over asking for a list of quotes**, which was the first design:

1. **Verification is exact, not fuzzy.** A quote-based scheme has to string-match the
   model's quote against the source, and models routinely swap a curly apostrophe or
   collapse a newline — so you need normalisation, and normalisation means a paraphrase can
   slip through while a good citation gets flagged. A line locator either resolves or it
   does not. No tolerance, no false verdicts either way.
2. **It resolves to a sentence, not a paragraph.** Which is exactly the granularity the UI
   needs to highlight, and the granularity a reader wants.
3. **Placement carries meaning.** The marker sits where the model put it, so a reader sees
   *which clause* is sourced, not just that the paragraph has references somewhere.
4. **Failure is visible.** A citation naming a label that was never supplied, or a line past
   the end of a passage, is kept but rendered `⚠ unverified`. The system reports its own
   ungroundedness instead of hiding it. Every report prints a grounding rate; live runs on
   this corpus produce **100% of claims resolving to a specific source line**.

### In the UI

Clicking a citation chip scrolls its evidence card into view and highlights the exact
sentence. The mapping is computed server-side from resolved line numbers — the browser does
no text searching — so it is deterministic and unit-tested. Cited lines are also
pre-highlighted, so the shape of the evidence is visible before any click.

---

## 4. Results

### End-to-end, live, on the supplied archive

Six labelled questions, `eval/goldens_case.json`. Reproduce with
`python -m detective eval --goldens eval/goldens_case.json --pipeline`.

| strategy | R-precision | recall@k | MRR | **distractor rate** |
|---|---|---|---|---|
| dense | 0.83 | 0.83 | 0.83 | 0.11 |
| lexical (BM25) | 0.42 | 0.42 | 0.58 | 0.33 |
| hybrid (RRF) | 0.75 | 0.83 | 0.75 | 0.11 |
| hybrid + rerank | 0.58 | 0.75 | 0.72 | 0.28 |
| pipeline:single | 0.42 | 0.50 | 0.50 | 0.18 |
| **pipeline:agentic** | 0.42 | 0.42 | 0.50 | **0.00** |

**Read the last column.** Every retrieval-only strategy puts a known distractor in front of
the detective 11–33% of the time. The full agentic pipeline puts one there **never** — the
threshold plus the gatekeeper reject them outright, and no distractor reached a report in
any of the six questions.

**And read the first four columns honestly: dense retrieval alone out-ranks the hybrid.**
That was not my prediction. On short, clean, single-topic paragraphs `text-embedding-3-small`
is simply strong, while BM25 over 24 tiny chunks that share most of their vocabulary is
noisy, and equal-weight RRF let the weaker ranker drag down the stronger one. Measuring it
led to weighting the fusion 2:1 toward dense, which recovered R-precision from 0.58 to 0.75
— but dense alone is still ahead on this corpus, and the table says so.

The reranker likewise does **not** improve document ranking here (0.58 vs 0.75). Its value
is elsewhere and is real: it produces a *calibrated* 0–1 relevance score, which is what makes
an absolute cut-off meaningful. Fused RRF scores cannot do that — they are relative to the
result set. The threshold is the gate; the reranker is what makes the gate possible.

The pipeline modes trade recall for precision (0.42 vs 0.83): the gatekeeper is conservative
and `max_evidence` caps the set. **For this task that is the right trade** — a report built
on three solid passages beats one that also cites a file announcing its own irrelevance —
but it is a trade, and `rerank_threshold` / `max_evidence` are the dials.

### Reproducible benchmark (CI)

`eval/synthetic/` is a fictional 10-document archive with the same adversarial structure —
including three distractors that declare their own irrelevance — labelled in
`eval/goldens.json`. CI scores it with deterministic offline stand-ins, so the numbers move
only when the retrieval architecture changes, never because a model drifted. The table is
published to the GitHub Actions job summary on every run.

### Sample output

`How did the hacker launder the stolen funds?` (agentic, live) admits `case_5#2`,
`case_2#0`, `case_5#1` — Tornado Cash, the Solana wallet, and the transaction splitting —
and rejects, with reasons:

- `case_2#2` at 0.69 — *"generic and speculative … says the hacker 'likely' used an
  automated script"* (the **highest-ranked passage of all**, rejected on reasoning)
- `case_1#1` at 0.47 — *"concerns how the attacker gained access …, not how the stolen funds
  were laundered"*
- `case_4#1`, `case_6#2`, `case_1#0` — below the relevance threshold

That first rejection is the whole argument in one line: the best-ranked passage was not the
best evidence, and only a stage that *reads* it could tell.

---

## 5. Case-agnostic by construction

The brief requires no case-specific information in the prompts. All prompts live in one
module written for a generic "investigative analyst over a document corpus"; the corpus
directory is a runtime parameter. `tests/test_prompts.py` tokenises **both** corpora and
fails the build if any prompt contains their vocabulary, plus a second check for domain
entities. Swapping `data/` for another folder needs no code and no prompt edit.

---

## 6. Two bugs the live runs found

Worth recording, because both were invisible to the test suite:

**Vector cache poisoning.** The cache was keyed on corpus content plus the *configured model
name*. Running `--offline` leaves that name untouched while producing 256-dimensional
stand-in vectors, so a later live run loaded them and crashed on a 1536-dimensional query.
Fixed by adding `identity` to the `Embedder` protocol and keying the cache on the embedder
that actually produced the vectors — which eliminates the whole class of bug rather than the
instance.

**Citation markers leaking into prose.** `open_questions` was not passed through the citation
parser, so raw delimiters rendered as `citeE1L1-L1` on screen. Fixed by stripping markers
from every free-text field. Related: entries the model listed as "excluded" while also citing
them are now dropped, so the report cannot contradict itself in front of a reader.

---

## 7. Honest limitations

- **Six and five labelled questions.** Enough to catch a regression, too few for a
  confidence interval. Treat single-point differences as noise.
- **`dense_weight = 2.0` is tuned on this corpus.** It is a setting, documented as measured
  rather than principled, and should be re-measured on new data.
- **Recall is the agentic mode's weak spot.** It stops when the gatekeeper is satisfied,
  which on the access question meant reporting phishing (`case_1`) without the
  credential-stuffing precursor (`case_3`). Defensible, but a recall-oriented deployment
  would raise `max_evidence` and lower the threshold.
- **No live-API test in CI.** By design — CI must not depend on a key or a model's mood —
  but it does mean provider-contract drift is caught only when someone runs live.
- **Sentence splitting is a regex.** Fine for these files; abbreviations and decimals would
  need a real segmenter.
