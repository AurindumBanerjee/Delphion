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
            # ✅ Clean base name + unique identifier
            clean_name = os.path.basename(file.filename)
            filename = f"{uuid.uuid4()}_{clean_name}"
            file_path = f"/tmp/{filename}"  # ✅ Safe absolute path

            with open(file_path, "wb") as f:
                f.write(await file.read())

            paths.append(file_path)

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save file {file.filename}: {str(e)}"
            }

    try:
        collection = load_documents(
            paths,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if collection is None:
            return {"status": "embedding failed", "reason": "no chunks created"}

        return {
            "status": "embedding complete",
            "chunks": len(collection.get().get("ids", []))
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Embedding failed: {str(e)}"
        }
