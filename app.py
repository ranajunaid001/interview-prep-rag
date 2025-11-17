from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import os
import pandas as pd
from pinecone import Pinecone
from openai import OpenAI
import google.generativeai as genai

app = FastAPI()

# Global storage for goldens (temporary - use database in production)
GOLDEN_EXAMPLES = []

# Initialize clients
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Get or create Pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME", "interview-prep")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Match Pinecone's requirement
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
        model="text-embedding-3-small",
        input=text,
        dimensions=1024  # Specify 1024 dimensions
    )
    return response.data[0].embedding

# Serve the HTML page
@app.get("/")
def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# RAG chat endpoint
@app.post("/api/chat")
async def chat(msg: ChatMessage):
    try:
        # Get embedding for the question
        query_embedding = get_embedding(msg.message)
        
        # Search Pinecone for relevant chunks
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Extract context from results - NO THRESHOLD
        context_chunks = []
        for match in results['matches']:
            context_chunks.append(match['metadata']['text'])
        
        if not context_chunks:
            return {"response": "I don't have any relevant information about that in my knowledge base. Please upload relevant documents first."}
        
        # Build prompt for Gemini
        context = "\n\n".join(context_chunks)
        prompt = f"""You are an interview prep assistant. Answer the user's question based ONLY on the provided context.
        
Context from documents:
{context}

User Question: {msg.message}

Instructions:
- Answer ONLY using information from the context above
- If the answer isn't in the context, say "I don't have information about that in the provided documents"
- Be concise and interview-focused
- Use a supportive, coaching tone

Answer:"""
        
        # Get response from Gemini
        response = gemini_model.generate_content(prompt)
        
        return {"response": response.text}
        
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

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

# Upload goldens endpoint
@app.post("/api/upload-goldens")
async def upload_goldens(file: UploadFile = File(...)):
    try:
        # Read Excel file
        content = await file.read()
        df = pd.read_excel(content)
        
        # Validate columns
        required_columns = ['question', 'ideal_answer', 'intent', 'expected_document']
        if not all(col in df.columns for col in required_columns):
            return {"success": False, "error": f"Excel must have columns: {required_columns}"}
        
        # Convert to list of dictionaries
        goldens = df.to_dict('records')
        
        # For now, store in a global variable (later use a database)
        global GOLDEN_EXAMPLES
        GOLDEN_EXAMPLES = goldens
        
        return {
            "success": True,
            "message": f"Uploaded {len(goldens)} golden examples",
            "examples": len(goldens),
            "sample": goldens[0] if goldens else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Get golden examples
@app.get("/api/get-goldens")
async def get_goldens():
    return {
        "total": len(GOLDEN_EXAMPLES),
        "goldens": GOLDEN_EXAMPLES
    }

# Metrics endpoint
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
            "gemini_ready": True,
            "golden_examples_loaded": len(GOLDEN_EXAMPLES)
        }
    except Exception as e:
        return {"status": "unhealthy", "pinecone_connected": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
