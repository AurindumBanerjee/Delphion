# backend/generation.py

from transformers import pipeline
from typing import List

# Load the generation model once
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",  # Swap with your preferred model
    max_new_tokens=300,
    do_sample=False
)

def format_prompt(user_question: str, context_chunks: List[str], system_prompt: str = None) -> str:
    """
    Formats the prompt by including context and question in a consistent structure.
    """
    context = "\n\n".join(context_chunks)
    
    system_instruction = system_prompt or (
        "Answer the user's question using only the provided context. "
        "If the answer is not in the context, say so clearly."
    )

    return f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {user_question}"


def generate_answer(user_question: str, context_chunks: List[str], system_prompt: str = None) -> str:
    """
    Generate an answer using the model, given the user question and context.
    """
    prompt = format_prompt(user_question, context_chunks, system_prompt)
    output = generator(prompt, return_full_text=False)
    return output[0]["generated_text"].strip()
