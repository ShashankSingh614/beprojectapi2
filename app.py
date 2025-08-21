from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nyayaFunction import modelRun
import os
from typing import Dict

app = FastAPI(
    title="Nyaya Chatbot API",
    description="API for querying the Bharatiya Nyaya Sanhita (BNS) dataset with semantic search and natural language responses.",
    version="1.0.0"
)

# Configure CORS (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  # e.g., ["https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post(
    "/chat/",
    summary="Process a user query",
    description="Receives a user query and returns a response based on the Bharatiya Nyaya Sanhita (BNS) dataset."
)
async def chat(chat_request: ChatRequest) -> Dict:
    try:
        if not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        response = modelRun(chat_request.message)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get(
    "/",
    summary="Check API status",
    description="Returns a message indicating that the Nyaya Chatbot API is running."
)
def root() -> Dict:
    return {"message": "Nyaya Chatbot API is running."}