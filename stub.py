import modal

app = modal.App("delphion-mcp")

GITHUB_REPO = "https://github.com/AurindumBanerjee/Delphion.git"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        f"git clone {GITHUB_REPO}",
        "pip install -r Delphion/requirements.txt"
    )
)

@app.function(
    image=image,
    timeout=300,
)
@modal.concurrent(max_inputs = 100)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/Delphion")

    from fastapi import FastAPI
    from mcp.embedder import app as embedder_app
    from mcp.generator import app as generator_app
    from mcp.retriever import app as retriever_app

    api = FastAPI()
    api.mount("/embed", embedder_app)
    api.mount("/generate", generator_app)
    api.mount("/retriever", retriever_app)

    return api


@app.local_entrypoint()
def main():
    print("Visit:", fastapi_app.get_web_url())
