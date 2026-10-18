# Archived versions

Kept for the record. Neither is maintained or wired into the current app.

| Version | Folder | Docs |
|---|---|---|
| v1 — Gradio monolith | [`v1-gradio-monolith/`](v1-gradio-monolith/) | [README](v1-gradio-monolith/README.md) · [BREAKDOWN.html](v1-gradio-monolith/BREAKDOWN.html) |
| v2 — Modal service split | [`v2-modal-mcp/`](v2-modal-mcp/) | [README](v2-modal-mcp/README.md) · [BREAKDOWN.html](v2-modal-mcp/BREAKDOWN.html) |

Each folder carries a `README.md` (file-by-file technical reference) and a
`BREAKDOWN.html` (annotated walkthrough with architecture diagrams — open it in
a browser).

## The arc in three lines

- **v1** proved the idea worked, and cost money on every upload and every question.
- **v2** split it into services to make the expensive parts scalable, and lost its
  vector index between containers in the process.
- **v3** (repository root) went back to one process with a session-scoped in-memory
  index and a local embedding model — v1's simplicity, without v1's bill or v2's bug.

## The one lesson worth carrying

v2 failed because **stateful storage was placed inside stateless compute**.
Chroma wrote to container-local disk while Modal autoscaled disposable
containers, so ingestion and retrieval could land in different places. No amount
of application-level fixing would have helped; the architecture guaranteed it.

The current version resolves this by construction: the index is explicitly
in-memory, explicitly scoped to one session, and never expected to outlive the
process that built it.
