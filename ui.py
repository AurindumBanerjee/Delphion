import gradio as gr
import requests

MCP_API = "https://<your-app-name>--delphion-mcp-fastapi_app.modal.run"

def upload_pdfs(files):
    files_payload = [("files", (f.name, open(f.name, "rb"), "application/pdf")) for f in files]
    resp = requests.post(f"{MCP_API}/embed", files=files_payload)
    return resp.json()

def ask_question(message):
    resp = requests.post(f"{MCP_API}/retriever", json={"input": message})
    return resp.json()["output"]

with gr.Blocks() as demo:
    gr.Markdown("### 💬 Delphion RAG System (MCP Agents)")
    
    files = gr.Files(file_types=[".pdf"], label="Upload PDFs")
    upload_btn = gr.Button("Embed Documents")
    upload_status = gr.Textbox(label="Upload Result")

    msg = gr.Textbox(label="Your Question")
    answer = gr.Textbox(label="Answer")

    upload_btn.click(upload_pdfs, inputs=files, outputs=upload_status)
    msg.submit(ask_question, inputs=msg, outputs=answer)

demo.launch()
