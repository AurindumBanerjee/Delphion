---
title: Delphion
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: Ask questions across your PDFs, with cited answers.
---

# Delphion

Retrieval-Augmented Generation over PDFs. Upload documents, ask questions, get
answers that cite the passage and page they came from.

Runs on a free CPU Space: embeddings and reranking are computed locally, the
index is in-memory, retrieval is hybrid (BM25 + dense, fused and reranked), and
answers stream from any OpenAI-compatible endpoint.

## How it works

```
PDF ──PyMuPDF──> page Documents ──chunker──> chunks
                                               │
                         dedupe ──┬── fastembed bge-small ──> Chroma (dense)
                                  └── tokenize ─────────────> BM25 (lexical)
                                               │
question ──> dense top-20 ─┐
         └─> BM25 top-20 ──┴─> Reciprocal Rank Fusion ──> cross-encoder rerank
                                                              │ top-k
                                                              ▼
                                         numbered passages ──> LLM ──stream──> cited answer
```

| Stage | Module | Notes |
|---|---|---|
| Extract | `src/extract_from_pdf.py` | PyMuPDF, one Document per page; attaches `source`/`page`/`title` for citations |
| Chunk | `src/chunking/` | three interchangeable strategies behind `ChunkingFactory` |
| Embed | `src/models.py` | `bge-small-en-v1.5` via fastembed (ONNX, no torch) |
| Index | `src/retrieval.py` | `HybridIndex`: Chroma (cosine) + BM25, content-hash dedupe, session-scoped |
| Retrieve | `src/retrieval.py` | `dense` / `bm25` / `hybrid` (RRF, k=60) / `hybrid_rerank` (ms-marco-MiniLM cross-encoder over top-20) |
| Answer | `src/rag.py` | numbered context passages, inline `[n]` citations, token streaming |

## Why hybrid retrieval

Dense embeddings match meaning but blur rare, exact terms — names, numbers,
jargon, archaic medical vocabulary. BM25 is the mirror image. Reciprocal Rank
Fusion merges the two rankings using ranks only (no score calibration needed),
and a cross-encoder then reads each (question, passage) pair together to pick
the final order. The reranker only sees 20 candidates, so it adds milliseconds,
not seconds. All four modes are selectable in the UI and measured below.

## Chunking strategies

- **recursive** — splits on paragraph → line → sentence → word boundaries. General default.
- **semantic** — embeds sentences and cuts where meaning shifts most (95th-percentile distance). Costs one embedding pass; free here because embeddings are local.
- **markdown** — splits on `#` headings, preserving section hierarchy. Only useful for documents that have them.

## Benchmark

The three chunking strategies exist to be compared, so there is a harness that
compares them:

```bash
python eval/run_eval.py
```

It sweeps method x chunk size x k over a question set whose ground truth is a
verbatim span (chunking-invariant, so one set measures every configuration) and
reports hit@k, MRR, precision@k and nDCG@k as a markdown table. Embeddings are
local, so the sweep is free and runs offline. See [`eval/README.md`](eval/README.md).

## Results

Corpus: three books (≈1,100 pages, 2.7M characters) — *Lessons from
Neuroscience for AI*, *The Society of Mind*, and Hahnemann's *Materia Medica*.
42 hand-written questions, each with a verbatim answer span verified against
the PDF (`eval/questions.json`).

Run it yourself: `run_benchmark.bat` (Windows) or `python eval/run_eval.py`.

<!-- BENCHMARK:START -->
_Pending first run — `run_benchmark.bat` fills this in automatically._
<!-- BENCHMARK:END -->

**Caveat on the question set.** Questions were written from sampled passages
and, despite paraphrasing, share key terms with their answers. That favours
lexical matching, so BM25's numbers here are an upper bound for it, not a
general claim that BM25 beats embeddings.

## Configuration

Copy `.env.example` to `.env` (locally) or set these as Space secrets:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | Groq | any OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | check your provider's current model list |
| `LLM_API_KEY` | — | not needed for local Ollama/vLLM |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | changing it invalidates any existing index |
| `DEFAULT_K` | `4` | passages retrieved per question |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `dense`, `bm25`, `hybrid`, or `hybrid_rerank` |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | any fastembed cross-encoder |
| `RERANK_CANDIDATES` | `20` | fused candidates passed to the reranker |

Retrieval works without an LLM key — you will get the passages, just not a
generated answer.

## Getting a Groq key

Generation needs an LLM endpoint; Groq's free tier is the default.

1. Go to <https://console.groq.com> and sign in (Google or GitHub works).
2. Open **API Keys** → **Create API Key**, name it `delphion`, and copy it
   (it starts with `gsk_` and is shown only once).
3. Locally: `copy .env.example .env` and set `LLM_API_KEY=gsk_...`.
   On a Space: **Settings → Variables and secrets → New secret**, name
   `LLM_API_KEY`.
4. If `llama-3.3-70b-versatile` has been retired, pick a current model from
   <https://console.groq.com/docs/models> and set `LLM_MODEL`.

Never commit `.env` — it is already in `.gitignore`.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # add your key
python app.py
```

## Tests

```bash
pip install pytest && pytest -q
```

The suite uses offline fakes (hashed embeddings, word-overlap reranker), so it
runs in CI without downloading models. GitHub Actions runs lint + tests on every
push.

## Repository history

Earlier iterations are kept under `legacy/` for reference:

- `legacy/v1-gradio-monolith/` — the original single-file Gradio app using the OpenAI API
- `legacy/v2-modal-mcp/` — the FastAPI microservice split (embedder/retriever/generator) deployed on Modal

Each folder has a `README.md` (file-by-file reference) and a `BREAKDOWN.html`
(annotated walkthrough with diagrams). `legacy/v2-modal-mcp/` holds the complete
v2 source (services, `src/`, UI, requirements), and the git tag `v2-modal-mcp`
points at the last commit where v2 was the repository root.

Both are archived, not maintained. The Modal version stored its Chroma index on
container-local disk while Modal autoscaled containers, so ingestion and
retrieval could land in different containers and retrieval would silently return
nothing. Collapsing to a single process with an in-memory index removed that
class of bug by construction.

## License

MIT
