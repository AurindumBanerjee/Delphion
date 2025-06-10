---
title: Delphion
app_file: ui.py
sdk: gradio
sdk_version: 5.33.0
---
# Delphion

Delphion is a modular Retrieval-Augmented Generation (RAG) system based on the **Model Context Protocol (MCP)**. It integrates LLM-based agents (retriever, embedder, generator), document ingestion, vector search, and a testable Gradio UI.

---

## 🚀 Deployment with Modal

You can deploy the entire system on [Modal](https://modal.com/) using just:

```bash
modal serve stub.py
```

This will:

- Clone the Delphion GitHub repo into the Modal container
- Install dependencies via `requirements.txt`
- Start a FastAPI app exposing all three MCP tools:
  - `POST /embed/embed` – document embedding
  - `POST /retriever/retriever` – semantic search
  - `POST /generate/generate` – prompt-based response generation

---

## 🛡️ Set Your API Key (securely)

For production or Modal cloud deployment:

1. Store your OpenAI API key securely:

```bash
modal secret create delphion-secrets OPENAI_API_KEY= "Insert your key here"
```

2. `stub.py` will automatically read it using:

```python
os.environ["OPENAI_API_KEY"]
```

For local development, use a `.env` file in the root:

```env
OPENAI_API_KEY=sk-...
```

---

## 🧪 Testing with Gradio UI

You can run the UI client locally to test all 3 agents:

1. Edit the base URL in `test.py`:

```python
BASE_API = "http://localhost:8000"  # or your deployed URL
```

2. Run:

```bash
python test.py
```

3. The UI lets you:
- 📎 Upload PDFs to `/embed/embed`
- ❓ Ask questions to `/retriever/retriever`
- 🧠 Send prompts to `/generate/generate`

---

## 🧠 Architecture Overview

```text
User Query
   │
   ▼
Gradio UI / API Client
   │
   ▼
[Modal Deployed FastAPI]
   ├──▶ /embed/embed      → vector index via FAISS/Chroma
   ├──▶ /retriever/retriever → semantic search
   └──▶ /generate/generate → LLM generation via OpenAI
```

---

## 🧱 Technologies Used

- 🧠 OpenAI (LLMs + embeddings)
- 🦜 LangChain + ChromaDB (vectorstore)
- 🧬 FastAPI (agent APIs)
- 🖼️ Gradio (testing UI)
- 🧊 Modal (GPU / cloud backend)
- 📄 PDF ingestion with PyMuPDF

---

## 🧰 Run Locally (Advanced Dev Only)

You can also run the services individually:

```bash
# Embedder
uvicorn mcp.embedder:app --port 8001

# Retriever
uvicorn mcp.retriever:app --port 8002

# Generator
uvicorn mcp.generator:app --port 8003
```

Then adjust the `test.py` URLs accordingly.

---

## 📜 License

MIT License. Attribution required for public use or distribution.

---

## 📣 Credits

Delphion is inspired by the Oracle of Delphi — designed to extract structured knowledge from unstructured documents via LLM agents.
