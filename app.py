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
        
        # Search Pinecone for relevant DOCUMENT chunks only
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"type": "document"}  # Only search documents, not goldens
        )
        
        # Extract context from results
        context_chunks = []
        sources = []
        for match in results['matches']:
            context_chunks.append(match['metadata']['text'])
            sources.append(match['metadata']['source'])
        
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
        
        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an interview prep assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {"response": response.choices[0].message.content, "sources": list(set(sources))}
        
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
                "id": f"doc_{file.filename}_{i}",
                "values": embedding,
                "metadata": {
                    "type": "document",
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
        
        # Store each golden in Pinecone
        vectors = []
        for i, golden in enumerate(goldens):
            # Create unique ID based on question
            golden_id = f"golden_{hash(golden['question']) % 1000000}"
            
            # Get embedding for the question
            embedding = get_embedding(golden['question'])
            
            vectors.append({
                "id": golden_id,
                "values": embedding,
                "metadata": {
                    "type": "golden",
                    "question": golden['question'],
                    "ideal_answer": golden['ideal_answer'],
                    "intent": golden['intent'],
                    "expected_document": golden['expected_document']
                }
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        
        return {
            "success": True,
            "message": f"Uploaded {len(goldens)} golden examples",
            "examples": len(goldens),
            "sample": goldens[0] if goldens else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Get golden examples from Pinecone
@app.get("/api/get-goldens")
async def get_goldens():
    try:
        # Query all goldens (using a dummy vector)
        dummy_embedding = [0.0] * 1024
        results = index.query(
            vector=dummy_embedding,
            top_k=10000,  # Get all goldens
            include_metadata=True,
            filter={"type": "golden"}
        )
        
        goldens = []
        for match in results['matches']:
            goldens.append({
                "question": match['metadata']['question'],
                "ideal_answer": match['metadata']['ideal_answer'],
                "intent": match['metadata']['intent'],
                "expected_document": match['metadata']['expected_document']
            })
        
        return {
            "total": len(goldens),
            "goldens": goldens
        }
    except Exception as e:
        return {"error": str(e)}

# Run evaluation endpoint
@app.post("/api/run-evaluation")
async def run_evaluation():
    try:
        # Get all goldens
        goldens_response = await get_goldens()
        goldens = goldens_response.get('goldens', [])
        
        if not goldens:
            return {"error": "No golden examples found"}
        
        evaluation_results = []
        
        for golden in goldens:
            # Get actual response from our system
            chat_response = await chat(ChatMessage(message=golden['question']))
            actual_answer = chat_response['response']
            sources = chat_response.get('sources', [])
            
            # Evaluate using Gemini as judge
            eval_prompt = f"""You are evaluating an interview prep assistant's response.

Golden Question: {golden['question']}
Expected Answer: {golden['ideal_answer']}
Expected Document: {golden['expected_document']}
Actual Answer: {actual_answer}
Sources Used: {sources}

Evaluate the response on two criteria:

1. ACCURACY: Is the actual answer factually correct based on the expected answer?
   - If it matches the key information: mark as "correct"
   - If it's wrong or hallucinated: mark as "incorrect"

2. TONE: Is the response empathetic and supportive like a good interview coach?
   - If it's warm, encouraging, and helpful: mark as "empathetic"
   - If it's cold, robotic, or harsh: mark as "not_empathetic"

Respond ONLY with this JSON format:
{{
    "accuracy": "correct" or "incorrect",
    "tone": "empathetic" or "not_empathetic",
    "explanation": "One short sentence why (max 15 words)"
}}"""
            
            # Get evaluation from Gemini
            eval_response = gemini_model.generate_content(eval_prompt)
            
            # Parse the evaluation
            try:
                eval_text = eval_response.text.strip()
                # Remove any markdown formatting
                if '```json' in eval_text:
                    eval_text = eval_text.split('```json')[1].split('```')[0]
                elif '```' in eval_text:
                    eval_text = eval_text.split('```')[1].split('```')[0]
                    
                eval_result = json.loads(eval_text)
            except:
                eval_result = {
                    "accuracy": "incorrect",
                    "tone": "not_empathetic",
                    "explanation": "Failed to parse evaluation"
                }
            
            evaluation_results.append({
                "question": golden['question'],
                "intent": golden['intent'],
                "expected_document": golden['expected_document'],
                "actual_answer": actual_answer,
                "sources": sources,
                "accuracy": eval_result['accuracy'],
                "tone": eval_result['tone'],
                "explanation": eval_result.get('explanation', '')
            })
        
        # Calculate metrics
        accuracy_correct = sum(1 for r in evaluation_results if r['accuracy'] == 'correct')
        tone_empathetic = sum(1 for r in evaluation_results if r['tone'] == 'empathetic')
        total = len(evaluation_results)
        
        return {
            "success": True,
            "total_evaluated": total,
            "metrics": {
                "accuracy": {
                    "correct": accuracy_correct,
                    "incorrect": total - accuracy_correct,
                    "pass_rate": accuracy_correct / total if total > 0 else 0
                },
                "tone": {
                    "empathetic": tone_empathetic,
                    "not_empathetic": total - tone_empathetic,
                    "pass_rate": tone_empathetic / total if total > 0 else 0
                }
            },
            "detailed_results": evaluation_results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Metrics endpoint - now gets real data
@app.get("/api/metrics")
async def get_metrics():
    # For now, return last evaluation results
    # In production, store these in database
    return {
        "message": "Run /api/run-evaluation first to generate metrics"
    }

# Health check for connections
@app.get("/api/health")
async def health_check():
    try:
        stats = index.describe_index_stats()
        
        # Count goldens
        dummy_embedding = [0.0] * 1024
        golden_results = index.query(
            vector=dummy_embedding,
            top_k=1,
            include_metadata=True,
            filter={"type": "golden"}
        )
        golden_count = len(golden_results['matches'])
        
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "total_vectors": stats.total_vector_count,
            "golden_examples": golden_count,
            "gemini_ready": True
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
