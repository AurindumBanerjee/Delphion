"""Chunking strategies.

All three run on the local embedding model, so semantic chunking no longer
needs an API key. Keeping three comparable strategies is deliberate: they are
the variable in the retrieval benchmark.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger


def recursive_chunk_documents(documents, chunk_size=1000, chunk_overlap=200, **_):
    """Split on paragraph -> line -> sentence -> word boundaries, in that order.

    Falls back down the separator list only when a chunk is still too large, so
    it keeps natural structure intact wherever it can. Good general default.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"recursive: {len(documents)} docs -> {len(chunks)} chunks")
    return chunks


def semantic_chunk_documents(documents, chunk_size=500, chunk_overlap=50, **_):
    """Embed sentences and cut where consecutive meaning shifts most.

    Pre-splits recursively first to bound the number of embedding calls.
    Degrades to those base chunks if the semantic pass fails for any reason.
    """
    base_chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(documents)

    try:
        from langchain_experimental.text_splitter import SemanticChunker

        from src.models import get_embeddings

        splitter = SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
        )
        chunks = splitter.split_documents(base_chunks)
        logger.info(f"semantic: {len(documents)} docs -> {len(chunks)} chunks")
        return chunks
    except Exception as exc:
        logger.error(f"Semantic chunking failed ({exc}); using recursive chunks")
        return base_chunks


def markdown_header_chunk_documents(documents, chunk_size=2000, chunk_overlap=200, **_):
    """Split on Markdown headings, preserving section hierarchy in metadata.

    Only useful for text that actually has '#' headings; chunks without any
    are passed through untouched.
    """
    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    base_chunks = recursive_chunk_documents(documents, chunk_size, chunk_overlap)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    chunks: List[Document] = []
    for doc in base_chunks:
        try:
            splits = splitter.split_text(doc.page_content)
            if not splits:
                chunks.append(doc)
                continue
            for split in splits:
                merged = dict(doc.metadata)
                merged.update(split.metadata)
                split.metadata = merged
            chunks.extend(splits)
        except Exception as exc:
            logger.warning(f"Header split failed ({exc}); keeping original chunk")
            chunks.append(doc)

    logger.info(f"markdown: {len(documents)} docs -> {len(chunks)} chunks")
    return chunks
