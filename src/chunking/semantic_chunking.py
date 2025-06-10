from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os 
from dotenv import load_dotenv
load_dotenv()


def semantic_chunk_documents(documents):
    '''Chunk documents using langchain and langchain_experimental.'''
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts1 = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents and split them into {len(texts1)} chunks, using CharacterTextSplitter.")
    print(f"Chunks: {texts1}")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type='percentile'
    )
    texts2 = text_splitter.split_documents(texts1)
    print(f"Loaded {len(documents)} documents and split them into {len(texts2)} chunks, using SemanticChunker.")
    print(f"Chunks: {texts2}")
    return texts2