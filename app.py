from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import os
from pinecone import Pinecone
from openai import OpenAI
import anthropic

app = FastAPI()

# Initialize clients
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Get or create Pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME", "interview-prep")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric='cosine'
    )
index = pc.Index(index_name)

# Model for chat messages
class ChatMessage(BaseModel):
    message: str

# Serve the HTML page
@app.get("/")
def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Test chat endpoint (keeping for now)
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

# Health check for Pinecone
@app.get("/api/health")
async def health_check():
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "vectors_count": stats.total_vector_count
        }
    except:
        return {"status": "unhealthy", "pinecone_connected": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Also update your `requirements.txt`:**
```
fastapi
uvicorn
anthropic
openai
pinecone-client
pandas
openpyxl
python-multipart
