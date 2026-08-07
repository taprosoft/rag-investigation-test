# Case Archive Investigator

A retrieval-augmented investigation system: point it at a folder of case files, ask a
question, and get a report where **every claim is footnoted to a specific sentence** of a
specific source — and where evidence that merely *sounds* relevant is rejected with a reason.

Built for the AI Detective Challenge. The eight supplied case files are in [data/](data/).

> **[REPORT.md](REPORT.md)** — the technical write-up: retrieval strategy and why, how the
> agentic loop works, the citation design, and measured results.

---

## Approach in one minute

The eight case files are adversarial by design: three of them state in their own text that
they have *no connection* to the theft, and all eight share the same security-and-finance
vocabulary. Finding relevant-looking text is easy. **Refusing text that is topically perfect
and evidentially worthless is the actual problem**, so the system is built around a precision
gate rather than a better ranker.

```
question
   │
   ├─ plan a query ──────────── LLM names the gap it is trying to close
   │
   ├─ hybrid search ─────────── dense embeddings + BM25, fused by reciprocal rank
   │
   ├─ rerank ───────────────── Cohere cross-encoder, scored against the question
   │
   ├─ threshold + shortlist ── absolute cut-off, max 2 passages per source file
   │
   ├─ gatekeeper ───────────── LLM admits / rejects each passage, with reasons,
   │                            then says whether the evidence is sufficient
   │                            └── not sufficient? loop back with the named gap
   │
   └─ synthesise ───────────── report with inline citation markers, each validated
                                against the passage and line it points at
```

Two modes, switchable at the CLI and in the UI:

| mode | what it does | cost |
|---|---|---|
| `single` | one hybrid search → rerank → threshold → synthesise | 1 LLM call |
| `agentic` *(default)* | the loop above: plan → verify → continue → stop | a few calls per round |

**Measured headline:** the agentic pipeline lets **zero** known distractors into a report,
against 11–33% for every retrieval-only strategy. Full numbers, including a result that
contradicted the original design hypothesis, are in [REPORT.md](REPORT.md#4-results).

---

## Setup

Python 3.12+.

```bash
python3 -m venv env
source env/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # then add OPENAI_API_KEY and COHERE_API_KEY
```

`.env` is gitignored and is never read by the test suite.

**Models used:** `text-embedding-3-small` (embeddings), `gpt-4o-mini` (planning,
gatekeeping, synthesis), Cohere `rerank-v3.5` (cross-encoder). All configurable in `.env`.

---

## Run it

### Web UI

```bash
python -m detective serve            # http://127.0.0.1:7860
```

Chat and report on the left, live evidence on the right. **Click any citation chip** — e.g.
`E1·L2` — and the exact source sentence lights up in the evidence panel. The retrieval trace
below the evidence shows every query tried and every passage rejected, with the reason.

### Command line

```bash
python -m detective ask "How did the hacker launder the stolen funds?"
python -m detective ask "How did the attacker first gain access?" --mode single --trace
python -m detective ask "..." --upload          # archive the report to S3
python -m detective ask "..." --corpus /other/case/folder
```

`--trace` prints the retrieval trace. `--offline` swaps in deterministic stand-ins and needs
no API keys — useful for exercising the machinery, not for judging answer quality.

### Benchmark

```bash
python -m detective eval                                    # synthetic set, offline-capable
python -m detective eval --goldens eval/goldens_case.json   # the supplied case files
python -m detective eval --goldens eval/goldens_case.json --pipeline   # + end-to-end modes
```

---

## Quality gates

```bash
ruff check . && ruff format --check .    # lint + format
mypy                                     # strict, src and tests
pytest                                   # 111 tests
```

All three run in CI on every push ([.github/workflows/ci.yml](.github/workflows/ci.yml)),
along with the retrieval benchmark, whose results are published to the job summary.

**The suite needs no API keys.** Every provider sits behind a `Protocol` with a
deterministic stand-in, so a green build means the logic is correct — not that a model
happened to behave that day.

---

## Layout

```
src/detective/
├── core/           # settings, domain types, corpus loading and chunking
├── retrieval/      # vector index, BM25, reciprocal rank fusion
├── providers/      # Protocols + OpenAI, Cohere, and deterministic stand-ins
├── investigation/  # prompts, citation markers, the agent loop and synthesis
├── evaluation/     # golden-set scoring harness
├── interfaces/     # Markdown and HTML rendering, Gradio app
└── storage/        # S3 report archival
eval/               # synthetic validation corpus + labelled query sets
tests/              # 111 tests, all offline
```

Two design notes worth knowing up front:

- **Prompts contain no case-specific knowledge.** They are written for a generic document
  archive, and a test fails the build if corpus vocabulary ever leaks into them. Point
  `--corpus` at a different folder and nothing else changes.
- **Vector and BM25 indexes are in-process**, which is the right call for 24 paragraphs —
  brute-force cosine is exact and instant. Both sit behind narrow interfaces, so moving to
  Qdrant/pgvector and OpenSearch is a one-class change. See
  [REPORT.md](REPORT.md#storage-in-memory-deliberately).

---

## S3 archival

Optional and disabled unless configured. Set `DETECTIVE_S3_BUCKET`, `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` in `.env`; reports upload to
`s3://<bucket>/reports/<timestamp>-<question-slug>.md` via the CLI's `--upload` flag or the
UI button, which stays disabled until credentials are present.
