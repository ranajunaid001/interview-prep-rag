from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json

app = FastAPI()

# Model for chat messages
class ChatMessage(BaseModel):
    message: str

# Serve the HTML page
@app.get("/")
def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Test chat endpoint
@app.post("/api/chat")
async def chat(msg: ChatMessage):
    # For now, just echo back
    return {"response": f"You said: {msg.message}"}

# Test upload endpoint
@app.post("/api/upload-goldens")
async def upload_goldens(file: UploadFile = File(...)):
    # For now, just return file info
    return {"filename": file.filename, "size": len(await file.read())}

# Test metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    # For now, return dummy data
    return {
        "total_evaluations": 0,
        "average_accuracy": 0.0,
        "average_tone": 0.0
    }
