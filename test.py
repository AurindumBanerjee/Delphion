# Test UI for Modal MCP Agent (Embedder + Retriever + Generator)
# Run it with `python test.py`

import gradio as gr
import requests
import os

# ✅ Replace this with your actual deployed Modal FastAPI base URL
MCP_API = os.getenv("MCP_API", "https://b23cs1006--delphion-mcp-fastapi-app-dev.modal.run")

def upload_pdfs(files):
    try:
        files_payload = [("files", (f.name, open(f.name, "rb"), "application/pdf")) for f in files]
        r = requests.post(f"{MCP_API}/embed", files=files_payload)
        return f"✅ Embedded {len(files)} file(s)\nChunks: {r.json().get('chunks')}"
    except Exception as e:
        return f"❌ Upload error: {e}"

def ask_question(q):
    try:
        r = requests.post(f"{MCP_API}/retriever", json={"input": q})
        return r.json().get("output", "❌ No response")
    except Exception as e:
        return f"❌ Retriever error: {e}"

def generate_response(prompt):
    try:
        r = requests.post(f"{MCP_API}/generate", json={"prompt": prompt})
        return r.json().get("output", "❌ No output from generator")
    except Exception as e:
        return f"❌ Generator error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## ✅ MCP Agent Test UI (Embedder + Retriever + Generator)")

    with gr.Row():
        pdfs = gr.Files(file_types=[".pdf"], label="📎 Upload PDFs")
        upload_output = gr.Textbox(label="Embedder Output", lines=2)
    gr.Button("Upload & Embed", variant="primary").click(upload_pdfs, inputs=pdfs, outputs=upload_output)

    with gr.Row():
        question = gr.Textbox(label="❓ Ask a Question", placeholder="What is LLaMA 3?")
        retriever_output = gr.Textbox(label="Retriever Answer", lines=5)
    question.submit(ask_question, inputs=question, outputs=retriever_output)

    with gr.Row():
        prompt = gr.Textbox(label="🧠 Raw Prompt to Generator", placeholder="Summarise the document")
        generator_output = gr.Textbox(label="Generator Output", lines=5)
    prompt.submit(generate_response, inputs=prompt, outputs=generator_output)

demo.launch()
