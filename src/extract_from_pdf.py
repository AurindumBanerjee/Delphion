"""PDF -> page Documents -> chunks, with citation metadata attached."""

import os
from typing import List

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from loguru import logger

from src.chunking.chunking_factory import ChunkingFactory


def load_and_chunk(
    pdf_paths: List[str],
    chunking_method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Return chunks whose metadata carries source/page/title, so answers can cite."""
    if not pdf_paths:
        logger.warning("No PDFs provided")
        return []

    chunker = ChunkingFactory.create_chunker(
        method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []

    for pdf_path in pdf_paths:
        try:
            pages = PyMuPDFLoader(pdf_path).load()

            with fitz.open(pdf_path) as doc:
                raw = doc.metadata or {}
            base = {
                "source": os.path.basename(pdf_path),
                "title": raw.get("title") or "",
                "author": raw.get("author") or "",
                "chunking_method": chunking_method,
            }

            chunks = chunker(pages)

            # Carry document-level metadata onto every chunk. PyMuPDFLoader
            # already sets a 0-indexed "page"; normalise it to 1-indexed.
            for chunk in chunks:
                page_no = chunk.metadata.get("page")
                chunk.metadata.update(base)
                chunk.metadata["page"] = (page_no + 1) if isinstance(page_no, int) else 1

            all_chunks.extend(chunks)
            logger.info(f"{base['source']}: {len(pages)} pages -> {len(chunks)} chunks")

        except Exception as exc:
            logger.error(f"Error processing {pdf_path}: {exc}")

    logger.info(f"Total {len(all_chunks)} chunks from {len(pdf_paths)} file(s)")
    return all_chunks
