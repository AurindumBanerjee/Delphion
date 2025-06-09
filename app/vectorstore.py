from langchain.vectorstores import Chroma
from app.embeddings import get_embedding_model
import os

DB_DIR = "./chroma_db"


def create_vectorstore(chunks):
    embeddings = get_embedding_model()
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)


def load_vectorstore():
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embedding_model())
