from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import nltk
from loguru import logger
import os
from dotenv import load_dotenv

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

def recursive_chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Recursive chunking that splits by paragraphs, then sentences, then characters.
    Better preserves document structure and context.
    
    This is one of the most effective general-purpose chunking methods.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks using RecursiveCharacterTextSplitter")
    return chunks

def semantic_chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Semantic chunking using OpenAI embeddings - breaks text into chunks based on
    semantic meaning rather than character count.
    
    This is considered state-of-the-art for maintaining semantic coherence.
    """
    # First create some base chunks with recursive method
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    base_chunks = base_splitter.split_documents(documents)
    
    # Use OpenAI embeddings for semantic analysis
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found. Falling back to recursive chunking.")
        return base_chunks
        
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type='percentile',
            add_start_index=True
        )
        
        chunks = semantic_splitter.split_documents(base_chunks)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks using semantic chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error in semantic chunking: {e}")
        logger.info("Falling back to recursive chunks")
        return base_chunks

def markdown_header_chunk_documents(documents, headers_to_split_on=None):
    """
    Splits documents based on Markdown headers, preserving document hierarchy.
    Especially useful for technical documentation, research papers, and reports.
    
    This method is excellent for documents with clear section headers.
    """
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    
    # First use recursive chunking to get base documents
    # This helps with documents that don't have many headers
    base_chunks = recursive_chunk_documents(documents, chunk_size=2000, chunk_overlap=200)
    
    # Then split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    chunks = []
    for doc in base_chunks:
        try:
            # Try to split by headers
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            # If we got no splits (no headers), keep the original chunk
            if not header_splits:
                chunks.append(doc)
                continue
                
            # Preserve the original metadata
            for split in header_splits:
                split.metadata.update(doc.metadata)
                # Add header info to metadata
                if "headers" in split.metadata:
                    for level, header in split.metadata["headers"].items():
                        split.metadata[f"header_{level}"] = header
            
            chunks.extend(header_splits)
        except Exception as e:
            logger.warning(f"Error splitting document by headers: {e}")
            # Fallback to keeping the original document
            chunks.append(doc)
    
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks using MarkdownHeaderTextSplitter")
    return chunks