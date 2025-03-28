from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from agents.graph import app as agent
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()



app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input schema
class ChatRequest(BaseModel):
    prompt: str
    thread_id: Optional[str] = None  # Auto-generate a unique ID if not provided

# Define output schema
class ChatResponse(BaseModel):
    thread_id: str
    response: str

# Initialize the agent
def chat(prompt: str, thread_id: str) -> str:
    """
    Chat function using the compiled graph application.

    Args:
        prompt (str): User input.
        thread_id (str): Thread ID for the session.

    Returns:
        str: Generated response.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [], "question": prompt}
    try:
        for output in agent.stream(inputs, config):
            for key, value in output.items():
                # Node details (debugging)
                pass

        # Final generation
        response: str = value["generation"]
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/AIvestor", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    FastAPI endpoint to interact with the agent.

    Args:
        request (ChatRequest): Contains user prompt and optional thread_id.

    Returns:
        ChatResponse: Response from the agent.
    """
    # Auto-generate thread_id if not provided
    thread_id = request.thread_id or str(uuid4())

    try:
        # Invoke the chat function
        response = chat(prompt=request.prompt, thread_id=thread_id)
        return ChatResponse(thread_id=thread_id, response=response)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    


if __name__ == "__main__":
    uvicorn.run(app)
