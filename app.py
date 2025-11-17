from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def home():
    return HTMLResponse(content="<h1>Interview Prep RAG</h1><p>System is running</p>")
