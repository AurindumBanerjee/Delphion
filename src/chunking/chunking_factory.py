from loguru import logger
from src.chunking.advanced_chunking import (
    recursive_chunk_documents,
    semantic_chunk_documents,
    markdown_header_chunk_documents
)

class ChunkingFactory:
    """Factory class to create document chunkers based on method name"""
    
    @staticmethod
    def get_chunking_methods():
        """Return a list of available chunking methods"""
        return {
            "recursive": "Recursive chunking by paragraphs, sentences, and characters - good for general use",
            "semantic": "Semantic chunking with OpenAI embeddings - best for preserving meaning and context",
            "markdown": "Chunk by Markdown headers - ideal for technical documents with clear sections"
        }
    
    @staticmethod
    def create_chunker(method="recursive", **kwargs):
        """
        Create a document chunker based on the specified method.
        
        Args:
            method (str): The chunking method to use
            **kwargs: Additional parameters to pass to the chunker
            
        Returns:
            function: A function that takes documents and returns chunks
        """
        chunkers = {
            "recursive": lambda docs: recursive_chunk_documents(docs, **kwargs),
            "semantic": lambda docs: semantic_chunk_documents(docs, **kwargs),
            "markdown": lambda docs: markdown_header_chunk_documents(docs, **kwargs),
        }
        
        if method not in chunkers:
            logger.warning(f"Unknown chunking method '{method}'. Falling back to recursive chunking.")
            method = "recursive"
            
        logger.info(f"Using {method} chunking method")
        return chunkers[method]