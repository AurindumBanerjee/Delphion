import os
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from loguru import logger

def get_chroma_collection(persist_directory: str = "./chroma_db"):
    """
    Returns a ChromaDB collection, creating it if needed.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    try:
        embedding_fn = OpenAIEmbeddings(model="text-embedding-3-large")
        return Chroma(
            collection_name="pdf_chunks",
            embedding_function=embedding_fn,
            persist_directory=persist_directory
        )
    except Exception as e:
        logger.error(f"Error getting Chroma collection: {e}")
        raise

def create_vectorstore(texts, metadatas, ids):
    """
    Creates or updates a vector store with the provided texts and metadata.
    
    Args:
        texts (list): List of text chunks to index.
        metadatas (list): List of metadata dictionaries.
        ids (list): List of unique IDs for each chunk.
        
    Returns:
        The Chroma collection object.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return None
        
    # Ensure directory exists
    os.makedirs("./chroma_db", exist_ok=True)
    
    try:
        # Create embedding function
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Create the vector store
        collection = Chroma.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            ids=ids,
            persist_directory="./chroma_db",
            collection_name="pdf_chunks"
        )
        
        # Ensure the changes are persisted
        collection.persist()
        logger.info("Vector store created and persisted successfully")
        return collection
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def inspect_vectorstore(collection=None):
    """
    Inspects the vector store and returns a report.
    
    Args:
        collection: Optional Chroma collection. If not provided, one will be loaded.
        
    Returns:
        A dictionary with statistics about the vector store.
    """
    try:
        if collection is None:
            collection = get_chroma_collection()
            
        if collection is None:
            return {"error": "No collection available"}
            
        # Get all documents
        all_docs = collection.get()
        
        if not all_docs or not all_docs.get('ids'):
            return {
                "total_chunks": 0,
                "status": "Vector store exists but contains no documents"
            }
            
        # Create report
        report = {
            "total_chunks": len(all_docs['ids']),
            "collection_name": "pdf_chunks",
            "status": "Vector store is available and contains documents"
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error inspecting vector store: {e}")
        return {"error": str(e)}