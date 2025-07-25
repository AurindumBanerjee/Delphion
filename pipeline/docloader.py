# pipeline/document_loader.py

import fitz  # PyMuPDF
import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document


def load_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load and parse PDF files into LangChain Document objects using PyMuPDF.
    
    Args:
        file_paths (List[str]): Paths to local PDF files.
    
    Returns:
        List[Document]: Parsed documents ready for chunking.
    """
    all_docs = []
    for path in file_paths:
        if not os.path.exists(path):
            continue
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
