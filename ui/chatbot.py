# ui/chatbot.py

import gradio as gr
from pipeline.document_loader import load_pdfs
from pipeline.chunking import get_chunker
from backend.embedding import create_chroma_collection
from backend.retrieval import retrieve_relevant_chunks
from backend.generation import generate_answer

import os
import uuid

collection_state = {"ready": False}


def process_pdfs(file_objs, method, size, overlap):
    if not file_objs:
        return "❌ No files uploaded.", gr.update(visible=False)

    paths = []
    for file in file_objs:
        file_path = f"./temp/{uuid.uuid4()}_{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.read())
        paths.append(file_path)

    docs = load_pdfs(paths)
    chunker = get_chunker(method)
    chunks = chunker(docs, chunk_size=size, chunk_overlap=overlap)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    create_chroma_collection(texts, metadatas, ids)
    collection_state["ready"] = True

    return f"✅ Processed {len(paths)} PDF(s) with {len(chunks)} chunks.", gr.update(visible=True)


def handle_question(question, chat_history, system_prompt):
    if not collection_state["ready"]:
        return chat_history + [[question, "❌ Please upload and process documents first."]], ""

    chunks = retrieve_relevant_chunks(question)
    context = [c["content"] for c in chunks]
    answer = generate_answer(question, context, system_prompt)
    return chat_history + [[question, answer]], ""
