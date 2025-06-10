import openai
from src.vectorstore import get_chroma_collection, inspect_vectorstore
import os
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI



def predict(message, history, collection, system_prompt):
    """
    Predict the response of the chatbot based on the user's message and the system prompt.
    
    Args:
        message (str): The user's message.
        history (list): The chat history.
        collection: The Chroma collection object.
        system_prompt (str): The system prompt for the chatbot.

    Returns:
        list: The updated chat history with the bot's response.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
    
    # Append user message to history
    history.append((message, None))
    
    # Check if collection exists
    if collection is None:
        # No documents have been uploaded yet
        reply = "Please upload PDF documents first before asking questions."
        history[-1] = (message, reply)
        return history
    
    try:
        # Get relevant context chunks from the collection
        context_chunks = collection.similarity_search(message, k=3)
        context_text = "\n\n".join([doc.page_content for doc in context_chunks])
        
        # Generate a response using the OpenAI API
        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {message}\n\nContext: {context_text}"}
            ]
        )
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": f"Question: {message}\n\nContext: {context_text}"}
        #     ]
        # )
        
        # Extract the assistant's reply
        reply = completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        reply = f"An error occurred while processing your question: {str(e)}"
    
    # Update the last entry in history with the bot's response
    history[-1] = (message, reply)
    
    return history