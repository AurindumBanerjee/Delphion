"""Delphion — RAG over your PDFs.

Single-process Gradio app, sized for a free Hugging Face Space:
  * embeddings run locally on CPU (no key, no cost)
  * the vector index is in-memory and scoped to your browser session
  * retrieval is hybrid: BM25 + dense vectors fused with RRF, then reranked
  * answers stream token by token
  * generation goes to any OpenAI-compatible endpoint you configure
"""

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from loguru import logger

from src import keys, models, theme, usage
from src.chunking.chunking_factory import ChunkingFactory
from src.config import DEFAULT_SYSTEM_PROMPT, settings
from src.extract_from_pdf import load_and_chunk
from src.rag import answer_stream
from src.retrieval import MODES
from src.vectorstore import build_vectorstore

load_dotenv()

METHODS = ChunkingFactory.get_chunking_methods()


SAMPLE_PDF = Path(__file__).parent / "samples" / "alice_in_wonderland.pdf"
# Checked against the sample: the first three surface the answer in every retrieval
# mode. The last two are where one method alone misses, so switching modes shows why
# hybrid helps: dense misses the "school lessons" answer, BM25 misses the "hurry" one.
SAMPLE_QUESTIONS = [
    "What did the jar Alice took from the shelf say on it?",
    "What does the Cheshire Cat leave behind when it vanishes?",
    "What was the Caterpillar smoking?",
    "What did Alice write about the Mock Turtle's school lessons?",
    "Why was the White Rabbit in a hurry?",
]
KEY_WARNING = (
    "> ⚠️ **No LLM key is configured on this Space.** Retrieval and the passages "
    "below still work, but answers need the `LLM_API_KEY` secret, or your own key "
    "under \"Use your own API key\"."
)
PRIVACY_NOTE = (
    "🔒 Your key is sent to this Space's server to make the request and is kept only in "
    "your session (in memory). It is never saved or logged. Use a restricted or "
    "throwaway key, and only on a Space you trust."
)
REFRESH_NOTE = (
    "Refresh makes one tiny request (1 token) to read the limits. For the shared key it "
    f"is limited to once every {usage.REFRESH_CACHE_SECONDS} s."
)


def panel_for(user_key, own_book):
    """The Usage panel text for whichever key this session is using."""
    if user_key:
        return usage.render_panel(True, own_book or usage.UsageBook())
    return usage.render_panel(False, usage.SHARED.snapshot())


def use_key(raw, user_key, own_book):
    res = keys.set_key(raw, user_key, own_book)
    # The textbox is always cleared: the key is never shown again, only its last 4 characters.
    return "", res.key, res.book, res.message, panel_for(res.key, res.book)


def drop_key():
    res = keys.remove_key()
    return "", None, None, res.message, panel_for(None, None)


def do_refresh(user_key, own_book):
    book, note = keys.refresh(user_key, own_book)
    panel = panel_for(user_key, book)
    return (f"{panel}\n\n_{note}_" if note else panel), book


def ingest(files, method, size, overlap, progress=gr.Progress()):
    """Upload handler: PDFs -> chunks -> in-memory index held in session state."""
    if not files:
        return None, "No files uploaded."
    return _index([f.name for f in files], method, size, overlap, progress=progress)


def load_sample(method, size, overlap, progress=gr.Progress()):
    """One click: index the bundled public-domain sample so the demo needs no upload."""
    return _index([str(SAMPLE_PDF)], method, size, overlap, label="the sample", progress=progress)


def _index(paths, method, size, overlap, label=None, progress=None):
    """Index PDFs, reporting real progress: reading, then embedding passage by passage."""
    step = progress or (lambda *args, **kwargs: None)
    try:
        step(0.02, desc="📖 Reading and splitting the PDFs…")
        chunks = load_and_chunk(paths, method, int(size), int(overlap))
        if not chunks:
            return None, "No text could be extracted. Is the PDF a scanned image?"

        total, done = len(chunks), 0

        def tick(n):  # called every few embedded passages, inside this request only
            nonlocal done
            done = min(done + n, total)
            step(0.12 + 0.83 * done / total, desc=f"✨ Embedding passages… {done} / {total}")

        step(0.12, desc=f"✨ Embedding {total} passages…")
        token = models.set_embed_progress(tick)
        try:
            store = build_vectorstore(chunks)
        finally:
            models.reset_embed_progress(token)
        step(1.0, desc="Done")
        return store, (
            f"✨ Indexed **{total} chunks** from "
            f"{label or f'{len(paths)} file(s)'} using *{method}* chunking. Ask away."
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


def ask(question, history, store, k, system_prompt, mode, user_key, own_book):
    """Generator: Gradio re-renders on every yield, so the answer streams in.

    `user_key` / `own_book` come from this browser session only. The key is passed
    explicitly to answer_stream; it is never put in settings, env, history or logs."""
    if not question.strip():
        yield history, "", "", panel_for(user_key, own_book), own_book
        return
    history = (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    sink = usage.Sink(own=bool(user_key), book=own_book)
    for partial, sources in answer_stream(
        store, question, int(k), system_prompt, mode, api_key=user_key or None, usage=sink
    ):
        history[-1]["content"] = partial
        book = sink.book if user_key else own_book
        yield history, "", sources, panel_for(user_key, book), book


with gr.Blocks(
    theme=theme.build_theme(), css=theme.build_css(), js=theme.FORCE_DARK_JS, title="Delphion"
) as demo:
    gr.Markdown(
        "# <span class='star-mark'>✦</span> Delphion\n"
        "Ask questions across your PDFs. Answers cite the passages they came from.\n\n"
        "*Documents are indexed in memory for your session only — nothing is stored.*"
    )

    if not settings.llm_configured:
        gr.Markdown(KEY_WARNING)

    store_state = gr.State(None)
    user_key_state = gr.State(None)    # this session's own key; never leaves session state
    own_usage_state = gr.State(None)   # this session's own usage record

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(label="Upload PDFs", file_types=[".pdf"])
            sample_btn = gr.Button("Load sample (Alice in Wonderland, public domain)")
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
            with gr.Accordion("Use your own API key", open=False):
                gr.Markdown(PRIVACY_NOTE)
                key_box = gr.Textbox(
                    label="API key", type="password", placeholder="Paste your key", max_lines=1
                )
                with gr.Row():
                    use_key_btn = gr.Button("Use my key")
                    remove_key_btn = gr.Button("Remove key")
                key_status = gr.Markdown(keys.in_use(None))
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
            gr.Examples(SAMPLE_QUESTIONS, inputs=question, label="Try with the sample")
            with gr.Accordion("Usage", open=True):
                usage_md = gr.Markdown(panel_for(None, None))
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm", scale=0)
                    gr.Markdown(REFRESH_NOTE)
            with gr.Accordion("Retrieved passages", open=True):
                sources = gr.Markdown("_Sources will appear here._")

    method.change(lambda m: f"*{METHODS[m]}*", method, method_info)

    files.upload(
        ingest,
        [files, method, size, overlap],
        [store_state, status],
        show_progress="full",
    )

    sample_btn.click(
        load_sample,
        [method, size, overlap],
        [store_state, status],
        show_progress="full",
    )

    ask_args = dict(
        fn=ask,
        inputs=[question, chatbot, store_state, top_k, system_prompt, mode,
                user_key_state, own_usage_state],
        outputs=[chatbot, question, sources, usage_md, own_usage_state],
    )
    question.submit(**ask_args)
    send.click(**ask_args)

    key_outputs = [key_box, user_key_state, own_usage_state, key_status, usage_md]
    use_key_args = dict(fn=use_key, inputs=[key_box, user_key_state, own_usage_state],
                        outputs=key_outputs)
    use_key_btn.click(**use_key_args)
    key_box.submit(**use_key_args)
    remove_key_btn.click(drop_key, None, key_outputs)
    refresh_btn.click(do_refresh, [user_key_state, own_usage_state], [usage_md, own_usage_state])
    # A fresh page load shows the current shared reading, not the one from server start.
    demo.load(lambda: panel_for(None, None), None, usage_md)


if __name__ == "__main__":
    demo.launch()
