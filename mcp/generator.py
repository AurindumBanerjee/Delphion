from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()

class GenInput(BaseModel):
    prompt: str

@app.post("/generate")
def generate(input: GenInput):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input.prompt}]
    )
    return {"output": completion.choices[0].message.content}
