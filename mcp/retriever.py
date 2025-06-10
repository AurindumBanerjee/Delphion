from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.vectorstore import get_chroma_collection

app = FastAPI()
collection = get_chroma_collection()

class RetrieverInput(BaseModel):
    input: str
    k: int = 4  # default top-k

@app.post("/retriever")
def retrieve(query: RetrieverInput):
    try:
        docs = collection.similarity_search(query.input, k=query.k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return {
            "output": "\n\n".join([d["content"] for d in results]),
            "chunks": results
        }
    except Exception as e:
        return {"error": str(e)}
