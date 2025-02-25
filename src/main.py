import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Get token limits from environment variables or use defaults
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 2000))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 1000))

# Initialize FastAPI app
app = FastAPI(title="Notif Chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=8000)  # Approx 2000 tokens
    max_tokens: Optional[int] = Field(default=1000, le=MAX_OUTPUT_TOKENS)  # Limit output tokens

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Print API key for debugging (last 4 chars)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        print("Using API key ending in:", api_key[-4:] if api_key else "None")

        # Validate message length
        if len(request.message) > 8000:  # Approx 2000 tokens
            raise HTTPException(
                status_code=400,
                detail=f"Message too long. Please limit to {MAX_INPUT_TOKENS} tokens (approximately 8000 characters)"
            )

        # Get response from Claude
        completion = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
            messages=[{"role": "user", "content": request.message}]
        )

        # Print the response for debugging
        print("Claude response:", completion)

        # Extract the response text
        response_text = completion.content[0].text if completion.content else "No response generated"

        return ChatResponse(
            response=response_text
        )

    except Exception as e:
        # Print the full error for debugging
        print("Error in chat endpoint:", str(e))
        error_msg = f"Chat error: {str(e)}"
        if "invalid x-api-key" in str(e).lower():
            error_msg = "Invalid API key. Please check your ANTHROPIC_API_KEY in the .env file."
        raise HTTPException(status_code=500, detail=error_msg)
