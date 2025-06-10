import gradio as gr
import requests
import os

# ✅ Replace this with your actual deployed Modal FastAPI base URL
BASE_API = os.getenv("MCP_API", "https://b23cs1006--delphion-mcp-fastapi-app-dev.modal.run")

EMBED_URL = f"{BASE_API}/embed/embed"
RETRIEVE_URL = f"{BASE_API}/retriever/retriever"
GENERATE_URL = f"{BASE_API}/generate/generate"

def upload_pdfs(files, method, size, overlap):
    try:
        files_payload = [("files", (f.name, open(f.name, "rb"), "application/pdf")) for f in files]
        data = {
            "chunking_method": method,
            "chunk_size": str(size),
            "chunk_overlap": str(overlap)
        }
        r = requests.post(EMBED_URL, files=files_payload, data=data)
        res = r.json()
        return f"✅ {res.get('status')}: {res.get('chunks', '?')} chunks"
    except Exception as e:
        return f"❌ Embed error: {e}"

def ask_question(q, k):
    try:
        r = requests.post(RETRIEVE_URL, json={"input": q, "k": k})
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
    gr.Markdown("## 📚 Delphion RAG Agent UI")

    with gr.Row():
        files = gr.Files(label="📎 Upload PDFs", file_types=[".pdf"])

        chunk_method = gr.Dropdown(choices=["recursive", "semantic"], value="recursive", label="Chunking Method")
        chunk_size = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size")
        chunk_overlap = gr.Slider(minimum=0, maximum=500, value=200, step=50, label="Chunk Overlap")

        embed_output = gr.Textbox(label="Embedder Output", lines=2)

    gr.Button("Run Embedder", variant="primary").click(
        upload_pdfs,
        inputs=[files, chunk_method, chunk_size, chunk_overlap],
        outputs=embed_output
    )

    with gr.Row():
        question = gr.Textbox(label="❓ Ask a Question")
        top_k = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Top-k Chunks")
        retriever_output = gr.Textbox(label="Retriever Output", lines=5)
        
    question.submit(ask_question, inputs=[question, top_k], outputs=retriever_output)

    with gr.Row():
        prompt = gr.Textbox(label="🧠 Prompt for Generator")
        generator_output = gr.Textbox(label="Generator Output", lines=5)
    prompt.submit(generate_response, inputs=prompt, outputs=generator_output)

demo.launch(share = True)
