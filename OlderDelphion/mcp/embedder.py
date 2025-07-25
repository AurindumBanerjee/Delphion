
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from src.extract_from_pdf import load_documents

app = FastAPI()

@app.post("/embed")
async def embed(
    files: list[UploadFile] = File(...),
    chunking_method: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    paths = []

    for file in files:
        try:
            clean_name = os.path.basename(file.filename)
            filename = f"{uuid.uuid4()}_{clean_name}"
            file_path = f"/tmp/{filename}"

            with open(file_path, "wb") as f:
                f.write(await file.read())

            paths.append(file_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save file {file.filename}: {e}"
            }

    try:
        collection = load_documents(
            paths,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        count = 0
        try:
            # Direct access to ChromaDB internal collection count
            count = collection._collection.count()
        except Exception:
            pass

        if count == 0:
            return {
                "status": "embedding failed",
                "chunks": 0,
                "reason": "no embeddings were stored"
            }

        return {
            "status": "embedding complete",
            "chunks": count
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Embedding failed: {e}"
        }
