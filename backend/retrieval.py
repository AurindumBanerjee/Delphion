# backend/retrieval.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import os

def load_chroma_collection(
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_chunks",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Chroma:
    """
    Load the persisted Chroma vector store with HuggingFace embeddings.
    """
    if not os.path.exists(persist_dir):
        raise ValueError(f"Vector store directory not found: {persist_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return db


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_chunks",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[dict]:
    """
    Retrieve top-k most relevant chunks from Chroma based on query.
    
    Returns a list of dicts with 'content' and 'metadata'.
    """
    db = load_chroma_collection(persist_dir, collection_name, model_name)
    results = db.similarity_search(query, k=top_k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]
