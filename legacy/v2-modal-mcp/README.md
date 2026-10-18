# v2 — Modal / FastAPI Service Split (archived)

The second iteration. The monolith was broken into three FastAPI applications —
**embedder**, **retriever**, **generator** — mounted under one ASGI app and
deployed to [Modal](https://modal.com), with a thin Gradio page talking to them
over HTTP.

> **Archived.** Not maintained, not deployed. The Modal URL referenced in
> `app.py` and `test.py` was a `modal serve` development URL and no longer exists.

> **Naming note.** The `mcp/` folder predates and has nothing to do with the
> Model Context Protocol. It stood for "micro-service components". The name is a
> real hazard — `mcp` is now a package on PyPI, and `from mcp.embedder import app`
> would shadow or be shadowed by it. Renaming it was on the list.

---

## What it did

Same RAG idea as v1, but each stage became an independently callable HTTP
endpoint, with the intent that they could scale and be reused separately.

## Architecture

```
  ┌──────────────────────────────────────┐
  │  app.py / test.py  (Gradio, local)   │
  │  requests.post(...) to three URLs    │
  └───────────────┬──────────────────────┘
                  │  HTTPS
                  ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Modal:  b23cs1006--delphion-mcp-fastapi-app-dev.run     │
  │                                                          │
  │  stub.py  @modal.asgi_app()                              │
  │    api = FastAPI()                                       │
  │    api.mount("/embed",     embedder_app)  ──▶ POST /embed/embed
  │    api.mount("/generate",  generator_app) ──▶ POST /generate/embed  (!)
  │    api.mount("/retriever", retriever_app) ──▶ POST /retriever/retriever
  │                                                          │
  │  image: debian_slim + git clone Delphion + uv pip install│
  │  secret: delphion-secret → OPENAI_API_KEY                │
  │  @modal.concurrent(max_inputs=100)                       │
  │                                                          │
  │  ┌────────────────┐   ┌────────────────┐                 │
  │  │  container A   │   │  container B   │   ...           │
  │  │  ./chroma_db   │   │  ./chroma_db   │                 │
  │  │  (populated)   │   │  (EMPTY)       │                 │
  │  └────────────────┘   └────────────────┘                 │
  └──────────────────────────────────────────────────────────┘
```

The two containers in that diagram are the story of this version. Everything
else worked.

---

## Files in this folder

### `stub.py` — the Modal deployment (51 lines)

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        f"echo rebuild-trigger-{modal.__version__}",   # forces rebuild
        "rm -rf Delphion",
        f"git clone {GITHUB_REPO}",
        "pip install uv",
        "uv pip install --system -r Delphion/requirements.txt",
    )
)
```

The application code was baked into the image **at build time by cloning from
GitHub**. Consequences:

- You could only deploy code you had already pushed to `main`. Local edits were
  invisible to the deployment.
- Docker layer caching meant an unchanged clone command reused a stale layer, so
  a fresh push produced no change — hence the `echo rebuild-trigger-...` hack to
  bust the cache.
- Every deploy was a full image rebuild and reinstall, making the edit → test
  loop minutes long.

`fastapi_app()` runs inside the container: it asserts `OPENAI_API_KEY` is present
(injected by `modal.Secret.from_name("delphion-secret")`), appends `/Delphion` to
`sys.path`, imports the three sub-apps, and mounts them.

`@modal.concurrent(max_inputs=100)` allowed 100 concurrent requests per container
and let Modal autoscale to more containers — the direct cause of the split-state
failure below. `timeout=60*60*24` kept a container alive up to a day.

**Note the doubled path segments.** The sub-app is mounted at `/embed` and its
own route is also `/embed`, so the real URL is `/embed/embed`. Both clients
hardcode this shape.

### `mcp/embedder.py` — ingestion (65 lines)

`POST /embed`, multipart: `files[]` plus form fields `chunking_method`,
`chunk_size`, `chunk_overlap`.

```python
clean_name = os.path.basename(file.filename)
filename = f"{uuid.uuid4()}_{clean_name}"
file_path = f"/tmp/{filename}"
```

`os.path.basename` strips any `../` path-traversal attempt from the client-supplied
filename, and the UUID prefix prevents collisions between the up-to-100
concurrent requests sharing one container's `/tmp`. This was the most defensive
code in the version.

It then calls `load_documents(...)` and reports the chunk count by reaching into
Chroma's private internals:

```python
count = collection._collection.count()
```

Wrapped in a bare `except Exception: pass`, so an upstream API change would
silently report `0` chunks and the response would claim `"embedding failed"`
regardless of whether embedding had worked.

### `mcp/retriever.py` — search (27 lines)

The smallest and cleanest file. A Pydantic `RetrieverInput(input: str, k: int = 4)`,
a `similarity_search`, and a response carrying both a joined `output` string and
a structured `chunks` list.

Its one flaw is on **line 6**:

```python
collection = get_chroma_collection()   # runs at import, i.e. container boot
```

The collection handle was bound once when the container started and never
refreshed, so even writes made by *the same container* after boot were not
guaranteed to be visible.

### `mcp/generator.py` — never implemented (35 lines)

**This file is a copy-paste of `embedder.py`.** It defines `POST /embed`, saves
uploads, and calls `load_documents`. It contains no LLM call of any kind.

Since it was mounted at `/generate`, the endpoint the UI actually called —
`POST /generate/generate` — did not exist and returned 404. The "Generator" box
in both clients therefore always showed `"No generator output"`.

It also regressed the embedder's filename handling, writing to
`/tmp/{file.filename}` with no `basename` and no UUID — reintroducing both the
traversal risk and the collision risk that `embedder.py` had fixed.

The result: **there was no retrieve → generate pipeline.** The user retrieved
passages into one textbox and would have had to paste them into another. The
"RAG" loop was never closed in this version.

### `app.py` — the client UI (74 lines)

Three `requests.post` calls against the hardcoded Modal URL, wired to three
buttons. Passes chunking options as form data. Reads `MCP_API` from the
environment with the dev URL as a default.

### `test.py` — earlier client (55 lines)

The same thing without the chunking controls. Superseded by `app.py`.

---

## Request trace: uploading a PDF

1. `app.py:upload_pdfs` opens each Gradio temp file and POSTs multipart to
   `{BASE_API}/embed/embed`.
2. Modal routes to *some* container; FastAPI dispatches to `embedder.embed`.
3. Each file is written to `/tmp/{uuid}_{name}`.
4. `load_documents(paths, ...)` →`PyMuPDFLoader` per file → page Documents.
5. `ChunkingFactory.create_chunker(method, chunk_size, chunk_overlap)` returns a
   lambda; calling it produces chunks.
6. `create_vectorstore(texts, metadatas, ids)` → `OpenAIEmbeddings`
   (`text-embedding-3-large`) → `Chroma.from_texts(persist_directory="./chroma_db")`.
7. `collection._collection.count()` → `{"status": "embedding complete", "chunks": N}`.

## Request trace: asking a question

1. `app.py:ask_question` POSTs `{"input": q, "k": k}` to `/retriever/retriever`.
2. Modal routes to *a container, possibly a different one*.
3. `retriever.retrieve` uses the module-level `collection` bound at that
   container's boot.
4. If that container never handled the upload, `./chroma_db` is empty and
   `similarity_search` returns `[]` → `{"output": ""}` → the UI shows nothing.

Step 4 is the defining failure of this version.

---

## Known issues (as archived)

1. **Split state — the fatal one.** Chroma persisted to container-local disk
   while Modal ran many disposable containers. Ingestion and retrieval could
   land in different containers. Nothing in the application code was wrong; the
   architecture guaranteed the failure.
2. **Import-time collection binding** in `retriever.py:6` compounded it.
3. **`generator.py` was never written** — no `/generate` endpoint existed.
4. **No retrieve → generate wiring.** The RAG loop was open.
5. **Build-time `git clone`** made local iteration impossible and every deploy
   slow, with a cache-busting hack to work around it.
6. **Ephemeral hardcoded URL.** `...-dev.modal.run` dies with `modal serve`;
   both clients hardcode it.
7. **Private-API chunk count** via `collection._collection`, silently swallowed.
8. **Filename handling regressed** in `generator.py`.
9. **`mcp/` shadows the PyPI `mcp` package.**
10. Inherited from v1: metadata discarded before reaching the store,
    `collection.persist()` raising on modern Chroma, `markdown` chunking crashing
    on forwarded kwargs, and full OpenAI dependency for both embedding and
    generation.

---

## What it got right

Worth recording, because not all of it was a mistake:

- The **service boundaries were sensible** — embed / retrieve / generate is the
  correct decomposition of a RAG system, and the retriever's dual
  string-plus-structured response was a good API design.
- **Secret injection via Modal** was the right way to handle the API key.
- The **embedder's filename sanitisation** showed real awareness of a class of
  bug most portfolio code ignores.
- Deploying at all — with a build image, a secret, and a public URL — was a
  meaningful step past "runs on my laptop".

The mistake was not the split. It was putting **stateful storage inside
stateless compute**, and that is the lesson the current version is built around.

---

## Why it was replaced

Because the bug was not fixable without changing one of two things: move the
index out of the container (a hosted vector DB or an attached volume), or stop
using many containers.

Given no budget and no need for persistence, the current version takes the second
path: one process, one in-memory index scoped to a session, on a free Hugging Face
Space. It also finishes the job this version started, by actually implementing
generation and wiring retrieval into it.

See [`../v1-gradio-monolith/README.md`](../v1-gradio-monolith/README.md) for what
came before, and `BREAKDOWN.html` in this folder for the annotated walkthrough.
