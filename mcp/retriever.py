from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.vectorstore import get_chroma_collection

app = FastAPI()
collection = get_chroma_collection()

class RetrieverInput(BaseModel):
    input: str

@app.post("/retriever")
def retrieve(query: RetrieverInput):
    docs = collection.similarity_search(query.input, k=3)
    return {"output": "\n\n".join([d.page_content for d in docs])}
