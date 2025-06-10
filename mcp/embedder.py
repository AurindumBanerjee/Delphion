from fastapi import FastAPI, UploadFile, File
from src.extract_from_pdf import load_documents

app = FastAPI()

@app.post("/embed")
async def embed(files: list[UploadFile] = File(...)):
    paths = []
    for file in files:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        paths.append(file_path)

    collection = load_documents(paths)
    return {"status": "embedding complete", "chunks": len(collection.get()['ids'])}
