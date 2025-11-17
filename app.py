from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import os
from pinecone import Pinecone
from openai import OpenAI
import google.generativeai as genai

app = FastAPI()

# Initialize clients
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

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

# Helper functions
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embedding(text):
    """Get OpenAI embedding for text"""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

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

# Document upload endpoint
@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Create embeddings and store in Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            vectors.append({
                "id": f"{file.filename}_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": file.filename,
                    "chunk_index": i
                }
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks from {file.filename}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

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

# Health check for connections
@app.get("/api/health")
async def health_check():
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "vectors_count": stats.total_vector_count,
            "gemini_ready": True
        }
    except:
        return {"status": "unhealthy", "pinecone_connected": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
