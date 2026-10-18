# v1 — Gradio Monolith (archived)

The first working version of Delphion. One Python process, one Gradio UI, one
in-memory reference to a Chroma collection. Everything — parsing, chunking,
embedding, retrieval, generation — happened inside a single request handler.

> **Archived.** This folder is a record, not a runnable app. The modules it
> imported (`src/extract_from_pdf.py`, `src/vectorstore.py`,
> `src/chunking/*`) stayed at the repository root and have since been rewritten
> for the current version. See [Running it today](#running-it-today).

---

## What it did

Upload one or more PDFs → the text is split into chunks → each chunk is embedded
with OpenAI's `text-embedding-3-large` → the vectors go into a Chroma collection
→ ask a question → the three nearest chunks are pasted into a `gpt-3.5-turbo`
prompt → the answer appears in a chat window.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │        single Python process            │
                    │                                         │
  PDF upload ──────▶│  ui.py : process_upload()               │
                    │      │                                  │
                    │      ▼                                  │
                    │  extract_from_pdf.load_documents()      │
                    │      │  PyMuPDFLoader → page Documents  │
                    │      ▼                                  │
                    │  ChunkingFactory.create_chunker()       │
                    │      │                                  │
                    │      ▼                                  │
                    │  vectorstore.create_vectorstore()       │
                    │      │  OpenAI embeddings (paid)        │
                    │      ▼                                  │
                    │  Chroma ──▶ ./chroma_db (on disk)       │
                    │      │                                  │
                    │      └──▶ gr.State(collection) ◀────┐   │
                    │                                     │   │
  question ────────▶│  ui.py : process_message()          │   │
                    │      │                              │   │
                    │      ▼                              │   │
                    │  predict.predict(msg, hist, ────────┘   │
                    │      │            collection, prompt)   │
                    │      │  similarity_search(k=3)          │
                    │      ▼                                  │
                    │  OpenAI gpt-3.5-turbo (paid)            │
                    │      │                                  │
  answer ◀──────────│──────┘                                  │
                    └─────────────────────────────────────────┘
```

The important structural fact: **the collection object lived in Gradio session
state**, not on disk. `./chroma_db` was written, but nothing ever read it back —
a page refresh meant re-uploading. This was invisible while it worked, and
became the whole problem in v2.

---

## Files in this folder

### `ui.py` — the entire application (139 lines)

A `gr.Blocks` layout with two closures wired to events.

| Element | Purpose |
|---|---|
| `collection_state = gr.State(None)` | holds the live Chroma object per browser session |
| `file_output.upload(...)` | fires `process_upload` when files land |
| `msg.submit` / `submit_btn.click` | both fire `process_message` |
| `chunking_method` radio | populated from `ChunkingFactory.get_chunking_methods()` |
| `system_prompt` textbox | user-editable, passed straight through to the API call |

`process_upload(files, method, c_size, c_overlap)` calls `load_documents` and
returns `(collection, status_string)` into `[collection_state, upload_status]`.
Returning the collection *as a Gradio output* is what put it in session state.

`process_message(message, history, collection, sys_prompt)` guards against empty
input, delegates to `predict`, and returns `(updated_history, "")` — the empty
string clears the textbox, a small idiom worth remembering.

The UI passed `chunk_size` and `chunk_overlap` from sliders, which meant
selecting the `markdown` method crashed: `markdown_header_chunk_documents` did
not accept those keyword arguments, but the factory forwarded them to every
chunker. The bug was never hit in practice because nobody chose that option.

### `predict.py` — retrieval + generation (67 lines)

```python
context_chunks = collection.similarity_search(message, k=3)
context_text = "\n\n".join([doc.page_content for doc in context_chunks])
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {message}\n\nContext: {context_text}"},
    ],
)
```

That is the whole RAG loop. Notes on it:

- **`k=3` is hardcoded.** The UI had no top-k control, so retrieval breadth was
  never tunable.
- **No citations were possible.** The chunks were flattened into one string with
  no source or page markers, so the default system prompt's instruction to
  "cite the relevant section or page" could not be satisfied. The metadata
  existed — `load_documents` built it — but was discarded before reaching Chroma.
- **`os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")` (line 24)**
  is a no-op when the key is set and raises `TypeError: str expected, not
  NoneType` when it isn't — so a missing key failed with a confusing error
  rather than a clear one.
- History used Gradio's old tuple format, `(user, bot)`, with `None` as a
  placeholder for the pending reply. Modern Gradio prefers a list of
  `{"role", "content"}` dicts.

### `chunking.py` — dead code (12 lines)

`character_chunk_documents()`, a bare `CharacterTextSplitter` at 500/50. Never
imported by the factory. Superseded before this version shipped.

### `semantic_chunking.py` — dead code (26 lines)

An earlier semantic chunker: `CharacterTextSplitter` first, then `SemanticChunker`
over the result, using `text-embedding-ada-002`. Also never imported — the
factory pulled `semantic_chunk_documents` from `advanced_chunking.py`, which used
`text-embedding-3-small`. Two functions with the same name, two different models,
one of them unreachable. Kept here because it shows the iteration.

---

## Dependencies

| Package | Used for |
|---|---|
| `gradio` | the entire UI |
| `pymupdf` / `PyMuPDFLoader` | PDF text extraction, one Document per page |
| `langchain`, `langchain-community` | Document type, loaders, text splitters |
| `langchain-experimental` | `SemanticChunker` |
| `langchain-chroma`, `chromadb` | vector store |
| `langchain-openai` | `OpenAIEmbeddings` |
| `openai` | `gpt-3.5-turbo` chat completions |
| `python-dotenv` | `OPENAI_API_KEY` from `.env` |
| `loguru` | logging |

**Every question cost money twice**: once to embed the query, once to generate.
Every upload cost money proportional to document length. This is the single
biggest reason the project moved on.

---

## Known issues (as archived)

1. **Metadata discarded.** `load_documents` carefully assembled per-chunk
   metadata (source, author, title, creation date, page) into `all_metadatas`,
   then passed LangChain's default metadata to the store instead. Citations were
   impossible as a result.
2. **`collection.persist()` removed upstream.** Newer `langchain-chroma` /
   Chroma dropped the method; the call raised, the surrounding `except` swallowed
   it, and `create_vectorstore` returned `None` — surfacing as
   "Failed to process documents" even when embedding had succeeded.
3. **`markdown` chunking raised `TypeError`** on the factory's forwarded kwargs.
4. **Retrieval was dense-only, top-3, unranked.** No keyword matching, no
   reranking, no way to measure whether it was any good.
5. **No persistence across refresh.** Disk was written, never read.
6. **Single-user by construction.** Fine locally; not deployable to more than
   one person without paying per query for all of them.

---

## Why it was replaced

It worked, and for a local demo it worked well. What it could not do was ship:
running it publicly meant paying OpenAI for every visitor's uploads and questions,
and the monolith gave no way to scale or separate the expensive parts.

**v2** answered that by splitting embedder / retriever / generator into separate
FastAPI services on Modal — which introduced a worse problem, since the Chroma
index went back to container-local disk while the containers multiplied. See
[`../v2-modal-mcp/README.md`](../v2-modal-mcp/README.md).

**v3** (the current root app) went back to a single process, but replaced OpenAI
embeddings with a local ONNX model and made the index explicitly in-memory and
session-scoped — keeping v1's simplicity while removing its cost and v2's
split-state bug.

---

## Running it today

This folder is not self-contained. To resurrect it you would need the
repository at commit `788681f` or earlier:

```bash
git checkout 788681f
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python ui/ui.py
```

It will still incur OpenAI charges, and `collection.persist()` will fail on any
recent Chroma. Read `BREAKDOWN.html` in this folder for the annotated walkthrough.
