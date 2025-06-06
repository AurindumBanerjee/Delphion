# Delphion

Delphion is a modular Retrieval-Augmented Generation (RAG) system designed for secure, high-performance knowledge retrieval and response generation. Inspired by the Oracle of Delphi, it leverages agent-based architecture using the Model Context Protocol (MCP), vector search, and GPU-accelerated compute from Nebius Cloud.

## Key Features

- Model Context Protocol (MCP) Tool Integration  
- Dynamic RAG with chunked document retrieval  
- PDF/Text ingestion with embedding pipeline  
- Modular FastAPI agents: Retriever, Generator, Memory  
- Vector store via FAISS or ChromaDB  
- Nebius Object Storage (S3-compatible) for documents  
- Deployable on Nebius Compute or Modal  
- Gradio-based demo UI or Next.js frontend  

## Architecture Overview

```
User Query  
   │  
   ▼  
[Frontend UI: Gradio / Next.js]  
   │  
   ▼  
[LLM: Claude / GPT-4 w/ MCP Tool Calling]  
   │  
   ├──▶ retriever_tool (FastAPI on Nebius)  
   │       └─ uses FAISS for semantic search  
   ├──▶ generator_tool (optional response synthesis)  
   └──▶ embedder_tool (document ingestion)  
```

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/delphion.git
cd delphion
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Retriever Agent

```bash
cd agents/retriever
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 4. Start the Gradio UI (optional)

```bash
cd frontend/gradio
python app.py
```

### 5. Load Documents & Build Vector Index

```bash
python tools/embedder.py --input ./docs --output ./vector_store
```

## Services Overview

| Service         | Endpoint        | Description                          |
|-----------------|-----------------|--------------------------------------|
| retriever_tool  | /retriever      | Searches vector DB for relevant info |
| generator_tool  | /generator      | Optional synthesis & summarization   |
| embedder_tool   | /embed          | Embeds docs to build vector index    |
| uploader_tool   | /upload         | Uploads docs to Nebius S3            |

## Tech Stack

- OpenAI / Claude (LLM)  
- FastAPI (MCP tools)  
- FAISS / ChromaDB (Vector store)  
- Python (3.10+)  
- Docker (for deployment)  
- Nebius Compute & Object Storage  
- Gradio / Streamlit (UI)  

## Security & Deployment

- All agents are stateless, containerized via Docker  
- Secure access via Nebius firewalls or NGINX proxy  
- API Keys stored via dotenv / secrets manager  

## Examples

Example input to the retriever_tool via HTTP:

```json
POST /retriever
{
  "input": "What is the Mamba architecture?"
}
```

Response:

```json
{
  "output": "The Mamba architecture is a selective state-space model for long-context reasoning..."
}
```

## Status

- Proof-of-concept Complete  
- UI & Agent Orchestration in progress  
- PDF ingestion and auto-indexing working  
- Ready for Hackathon deployment on Nebius  

## Project Name Meaning

Delphion is inspired by the Oracle of Delphi—where knowledge and prophecy met. Just like the ancient priestesses channeled insight from the divine, Delphion channels knowledge from structured memory via smart agents and LLMs.

## License

MIT License. Use freely with attribution.
****
