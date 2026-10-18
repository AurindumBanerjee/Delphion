"""Delphion — RAG over your PDFs.

Single-process Gradio app, sized for a free Hugging Face Space:
  * embeddings run locally on CPU (no key, no cost)
  * the vector index is in-memory and scoped to your browser session
  * retrieval is hybrid: BM25 + dense vectors fused with RRF, then reranked
  * answers stream token by token
  * generation goes to any OpenAI-compatible endpoint you configure
"""

import gradio as gr
from dotenv import load_dotenv
from loguru import logger

from src.chunking.chunking_factory import ChunkingFactory
from src.config import DEFAULT_SYSTEM_PROMPT, settings
from src.extract_from_pdf import load_and_chunk
from src.rag import answer_stream
from src.retrieval import MODES
from src.vectorstore import build_vectorstore

load_dotenv()

METHODS = ChunkingFactory.get_chunking_methods()


def ingest(files, method, size, overlap):
    """Upload handler: PDFs -> chunks -> in-memory index held in session state."""
    if not files:
        return None, "No files uploaded."
    try:
        paths = [f.name for f in files]
        chunks = load_and_chunk(paths, method, int(size), int(overlap))
        if not chunks:
            return None, "No text could be extracted. Is the PDF a scanned image?"
        store = build_vectorstore(chunks)
        return store, (
            f"Indexed **{len(chunks)} chunks** from {len(paths)} file(s) "
            f"using *{method}* chunking. Ask away."
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        return None, f"Ingestion failed: {exc}"


MODE_LABELS = {
    "hybrid_rerank": "Hybrid + rerank (best)",
    "hybrid": "Hybrid (BM25 + dense)",
    "dense": "Dense only",
    "bm25": "BM25 only",
}


def ask(question, history, store, k, system_prompt, mode):
    """Generator: Gradio re-renders on every yield, so the answer streams in."""
    if not question.strip():
        yield history, "", ""
        return
    history = (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    for partial, sources in answer_stream(store, question, int(k), system_prompt, mode):
        history[-1]["content"] = partial
        yield history, "", sources


with gr.Blocks(theme=gr.themes.Soft(), title="Delphion") as demo:
    gr.Markdown(
        "# Delphion\n"
        "Ask questions across your PDFs. Answers cite the passages they came from.\n\n"
        "*Documents are indexed in memory for your session only — nothing is stored.*"
    )

    store_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(label="Upload PDFs", file_types=[".pdf"])
            method = gr.Radio(
                choices=list(METHODS), value="recursive", label="Chunking method"
            )
            method_info = gr.Markdown(f"*{METHODS['recursive']}*")
            size = gr.Slider(250, 2000, value=1000, step=250, label="Chunk size")
            overlap = gr.Slider(0, 500, value=200, step=50, label="Chunk overlap")
            top_k = gr.Slider(1, 10, value=settings.default_k, step=1, label="Passages to retrieve")
            mode = gr.Dropdown(
                choices=[(MODE_LABELS[m], m) for m in MODES if m in MODE_LABELS],
                value=settings.retrieval_mode,
                label="Retrieval mode",
            )
            status = gr.Markdown("No documents indexed yet.")
            with gr.Accordion("System prompt", open=False):
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT, lines=6, show_label=False
                )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=460, type="messages")
            with gr.Row():
                question = gr.Textbox(
                    placeholder="What does the document say about…",
                    show_label=False,
                    scale=8,
                )
                send = gr.Button("Ask", variant="primary", scale=1)
            gr.ClearButton([question, chatbot], value="Clear chat")
            with gr.Accordion("Retrieved passages", open=True):
                sources = gr.Markdown("_Sources will appear here._")

    method.change(lambda m: f"*{METHODS[m]}*", method, method_info)

    files.upload(
        ingest,
        [files, method, size, overlap],
        [store_state, status],
        show_progress="full",
    )

    ask_args = dict(
        fn=ask,
        inputs=[question, chatbot, store_state, top_k, system_prompt, mode],
        outputs=[chatbot, question, sources],
    )
    question.submit(**ask_args)
    send.click(**ask_args)


if __name__ == "__main__":
    demo.launch()
