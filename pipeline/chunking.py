# pipeline/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def recursive_chunk(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    return splitter.split_documents(documents)


def semantic_chunk(
    documents: list[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> list[Document]:
    # Step 1: base chunks
    base_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    base_chunks = base_splitter.split_documents(documents)

    # Step 2: semantic chunking
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    semantic_splitter = SemanticChunker(embeddings)
    return semantic_splitter.split_documents(base_chunks)


def get_chunker(method: str = "recursive"):
    if method == "semantic":
        return semantic_chunk
    return recursive_chunk  # default
