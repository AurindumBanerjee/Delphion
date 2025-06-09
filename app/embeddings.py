import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_embedding_model(model_name: str = "openai"):
    """
    Return the embedding model based on the selected provider.
    Currently supports:
        - OpenAI
        - HuggingFace (e.g., BGE, MiniLM)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", model_name).lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    elif provider == "huggingface":
        model_path = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_path)

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
