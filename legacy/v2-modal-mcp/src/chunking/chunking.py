from langchain.text_splitter import CharacterTextSplitter
import os 
from dotenv import load_dotenv
load_dotenv()


def character_chunk_documents(documents):
    '''Chunk documents using langchain and langchain_experimental.'''
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts1 = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents and split them into {len(texts1)} chunks, using CharacterTextSplitter.")
    print(f"Chunks: {texts1}")
    return texts1