# Test UI for Modal MCP Agent (Embedder + Retriever + Generator)
# Run it with `python test.py`

import gradio as gr
import requests
import os

# ✅ Replace this with your actual deployed Modal FastAPI base URL
BASE_API = os.getenv("MCP_API", "https://b23cs1006--delphion-mcp-fastapi-app-dev.modal.run")

EMBED_URL = f"{BASE_API}/embed/embed"
RETRIEVE_URL = f"{BASE_API}/retriever/retriever"
GENERATE_URL = f"{BASE_API}/generate/generate"

def upload_pdfs(files):
    try:
        files_payload = [("files", (f.name, open(f.name, "rb"), "application/pdf")) for f in files]
        r = requests.post(EMBED_URL, files=files_payload)
        res = r.json()
        return f"✅ Embedded: {res.get('chunks', 'unknown')} chunks"
    except Exception as e:
        return f"❌ Embedder error: {e}"

def ask_question(q):
    try:
        r = requests.post(RETRIEVE_URL, json={"input": q})
        return r.json().get("output", "❌ No retriever response")
    except Exception as e:
        return f"❌ Retriever error: {e}"

def generate_response(prompt):
    try:
        r = requests.post(GENERATE_URL, json={"prompt": prompt})
        return r.json().get("output", "❌ No generator output")
    except Exception as e:
        return f"❌ Generator error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## 🧪 Test Your MCP Agent Stack")

    with gr.Row():
        pdfs = gr.Files(file_types=[".pdf"], label="📎 Upload PDFs")
        embed_output = gr.Textbox(label="Embedder Response", lines=2)
    gr.Button("Run Embedder", variant="primary").click(upload_pdfs, inputs=pdfs, outputs=embed_output)

    with gr.Row():
        question = gr.Textbox(label="❓ Ask Question", placeholder="e.g. What is Mamba?")
        retriever_output = gr.Textbox(label="Retriever Response", lines=5)
    question.submit(ask_question, inputs=question, outputs=retriever_output)

    with gr.Row():
        prompt = gr.Textbox(label="🧠 Generator Prompt", placeholder="e.g. Summarize this")
        generator_output = gr.Textbox(label="Generator Output", lines=5)
    prompt.submit(generate_response, inputs=prompt, outputs=generator_output)

demo.launch()