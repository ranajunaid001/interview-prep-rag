from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import os

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
    return {"response": f"You said: {msg.message}"}

# Test upload endpoint
@app.post("/api/upload-goldens")
async def upload_goldens(file: UploadFile = File(...)):
    return {"filename": file.filename, "size": len(await file.read())}

# Test metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    return {
        "total_evaluations": 0,
        "accuracy": {
            "correct": 0,
            "incorrect": 0,
            "pass_rate": 0.0
        },
        "tone": {
            "empathetic": 0, 
            "not_empathetic": 0,
            "pass_rate": 0.0
        }
    }

# This is important for Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Also create a `Procfile` in your root directory:**
```
web: python app.py
