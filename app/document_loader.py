from langchain.document_loaders import PyPDFLoader
import os


def load_pdfs(file_paths):

    """
    Given a list of PDF file paths, returns a combined list of LangChain Document objects.
    """

    all_docs = []

    for path in file_paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        
        
    return all_docs
