# app.py

import gradio as gr
from ui.chatbot import process_pdfs, handle_question

with gr.Blocks(title="📚 Delphion HF RAG") as demo:
    gr.Markdown("# 📚 Delphion (Hugging Face RAG)")
    gr.Markdown("Upload PDF files and ask questions about their content.")

    # Upload + chunking panel
    with gr.Row():
        files = gr.Files(label="📎 Upload PDFs", file_types=[".pdf"])
        method = gr.Radio(["recursive", "semantic"], value="recursive", label="Chunking Method")
        chunk_size = gr.Slider(250, 2000, step=250, value=1000, label="Chunk Size")
        chunk_overlap = gr.Slider(0, 500, step=50, value=200, label="Chunk Overlap")
    upload_btn = gr.Button("📤 Process PDFs")
    upload_status = gr.Textbox(label="Upload Status")

    gr.Markdown("---")

    # Chatbot interface
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Question", placeholder="Ask something...")
    system_prompt = gr.Textbox(
        label="System Prompt",
        value="You are a helpful assistant. Answer using only the PDF content. If unsure, say so."
    )
    submit_btn = gr.Button("Ask")
    clear_btn = gr.ClearButton([chatbot, msg], value="Clear Chat")

    # Bind logic
    upload_btn.click(
        fn=process_pdfs,
        inputs=[files, method, chunk_size, chunk_overlap],
        outputs=[upload_status, submit_btn]
    )
    submit_btn.click(
        fn=handle_question,
        inputs=[msg, chatbot, system_prompt],
        outputs=[chatbot, msg]
    )
    msg.submit(
        fn=handle_question,
        inputs=[msg, chatbot, system_prompt],
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch()
