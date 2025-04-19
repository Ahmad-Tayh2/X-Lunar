"""
FastAPI application to serve the faculty information chatbot as a REST API.
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import FacultyInfoChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faculty-chatbot")

# Initialize the FastAPI app
app = FastAPI(
    title="Faculty Information Chatbot API",
    description="API for querying information about faculty members",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the chatbot
chatbot = FacultyInfoChatbot()

# Define request and response models
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

# Define root endpoint to serve the web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    """Serve the web UI for the chatbot."""
    with open(os.path.join("static", "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Define endpoint for chatbot queries
@app.post("/ask", response_model=ChatResponse)
async def ask_question(chat_request: ChatRequest):
    """
    Ask a question about faculty information.
    
    Args:
        chat_request: The chat request containing the question and optional session ID
        
    Returns:
        A response containing the answer and session ID
    """
    logger.info(f"Received question: {chat_request.question}")
    logger.info(f"Session ID: {chat_request.session_id}")
    
    try:
        # Add timeout to prevent hanging
        try:
            # Get the answer from the chatbot with a timeout
            answer = await asyncio.wait_for(
                chatbot.ask(chat_request.question), 
                timeout=120.0
            )
            logger.info(f"Got answer: {answer[:50]}...")
            
            # Return the response
            return ChatResponse(
                answer=answer,
                session_id=chat_request.session_id or "default"
            )
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return JSONResponse(
                status_code=504,
                content={"error": "The request timed out. Please try again."}
            )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "service": "faculty_info_chatbot"}

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)