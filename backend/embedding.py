# backend/embedding.py

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from typing import List
import os

def create_embedding_function(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def create_chroma_collection(
    texts: List[str],
    metadatas: List[dict],
    ids: List[str],
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_chunks",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = create_embedding_function(model_name)

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    db.persist()
    return db
