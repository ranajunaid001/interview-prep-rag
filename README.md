# Interview Prep RAG System with AI Evaluation

## Overview

This system is an AI-powered interview preparation assistant that uses Retrieval-Augmented Generation (RAG) to provide accurate, document-based answers to interview questions. It includes a comprehensive evaluation system following the Anthropic Evaluations (AE vals) methodology to measure accuracy and tone quality.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  FastAPI Backend │────▶│   Pinecone DB   │
│   (index.html)  │     │    (app.py)      │     │  (Documents)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                │
                        ┌───────▼────────┐
                        │  PostgreSQL    │
                        │  (Goldens &    │
                        │   Eval Results)│
                        └────────────────┘
```

## Core Components

### 1. **Frontend Interface (`index.html`)**
- **Purpose**: Provides a minimalist, Jony Ive-inspired UI for users to interact with the system
- **Features**:
  - **Chat Tab**: Users can ask interview-related questions
  - **Upload Tab**: Interface for uploading golden examples (Excel files)
  - **Dashboard Tab**: View evaluation metrics and results
- **Design**: Clean, fast, efficient with smooth animations and sticky navigation

### 2. **Backend API (`app.py`)**
The FastAPI backend orchestrates all system operations:

#### **2.1 RAG Pipeline**
- **Document Processing**: 
  - Accepts `.txt` files with interview-related content
  - Chunks documents into 500-character segments with 50-character overlap
  - Generates embeddings using OpenAI's `text-embedding-3-small` model (1024 dimensions)
  - Stores document chunks in Pinecone vector database

- **Question Answering**:
  - Converts user questions to embeddings
  - Searches Pinecone for top 3 most relevant chunks
  - Sends context + question to OpenAI GPT-3.5-turbo
  - Returns grounded, factual answers based only on uploaded documents

#### **2.2 Evaluation System**
Following AE vals methodology with 4 steps:

1. **Goldens Management**: 
   - Accepts Excel files with columns: `question`, `ideal_answer`, `intent`, `expected_document`
   - Stores in PostgreSQL for persistence
   - Supports insert/update operations

2. **Synthetic Data**: 
   - System designed to support synthetic data generation (not yet implemented)
   - Would generate variations of golden examples for robust testing

3. **Human Grading**:
   - Golden examples serve as human-curated "perfect" answers
   - Each golden defines what "correct" looks like

4. **Auto-Rater (Judge)**:
   - Uses Google Gemini 2.5 Flash as an impartial judge
   - Evaluates each bot response on two dimensions:
     - **Accuracy**: Is the answer factually correct based on the golden?
     - **Tone**: Is the response empathetic and coach-like?
   - Generates binary pass/fail for each dimension

### 3. **Database Schema**

#### **PostgreSQL Tables**

**`goldens` table**:
```sql
CREATE TABLE goldens (
    id SERIAL PRIMARY KEY,
    question TEXT UNIQUE,        -- The test question
    ideal_answer TEXT,           -- Perfect answer
    intent VARCHAR(50),          -- Question category
    expected_document VARCHAR(100), -- Which doc should contain answer
    created_at TIMESTAMP DEFAULT NOW()
);
```

**`evaluation_runs` table**:
```sql
CREATE TABLE evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP DEFAULT NOW(),
    accuracy_rate FLOAT,         -- Percentage of accurate answers
    tone_rate FLOAT,            -- Percentage with good tone
    total_evaluated INTEGER,
    details JSONB               -- Full evaluation details
);
```

### 4. **Vector Storage (Pinecone)**
- **Index**: `interview-prep` (1024 dimensions, cosine similarity)
- **Document vectors**: 
  ```json
  {
    "id": "doc_Google_PM_Role.txt_0",
    "values": [0.123, -0.456, ...], // 1024-dim embedding
    "metadata": {
      "text": "Product Manager at Google...",
      "source": "Google_PM_Role.txt",
      "chunk_index": 0
    }
  }
  ```

## API Endpoints

### Core Functionality
- `GET /` - Serves the main UI
- `POST /api/chat` - RAG-powered chat endpoint
- `POST /api/upload-document` - Upload interview documents for RAG
- `POST /api/upload-goldens` - Upload golden examples (Excel)
- `GET /api/get-goldens` - Retrieve all golden examples
- `POST /api/run-evaluation` - Execute full evaluation suite
- `GET /api/metrics` - Get latest evaluation metrics
- `GET /api/health` - System health check

## Evaluation Flow

1. **User uploads golden examples** (Excel file with test Q&As)
2. **System runs evaluation**:
   ```
   For each golden question:
   ├── Send question to RAG chat
   ├── Get actual response from OpenAI GPT-3.5
   ├── Send to Gemini judge with:
   │   ├── Golden answer
   │   ├── Actual answer
   │   └── Expected document
   └── Gemini returns:
       ├── accuracy: "correct" or "incorrect"
       └── tone: "empathetic" or "not_empathetic"
   ```
3. **Results stored in PostgreSQL** with detailed breakdowns
4. **Metrics calculated**: Accuracy rate, Tone rate

## Tech Stack

### APIs & Services
- **OpenAI API**: 
  - GPT-3.5-turbo for answering questions
  - text-embedding-3-small for embeddings
- **Google Gemini API**: 
  - gemini-2.5-flash as evaluation judge
- **Pinecone**: Vector database for document storage
- **PostgreSQL**: Relational database for goldens and results

### Frameworks & Libraries
- **FastAPI**: Web framework
- **psycopg2**: PostgreSQL adapter
- **pandas**: Excel file processing
- **Railway**: Deployment platform

## Configuration

### Environment Variables
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=interview-prep
DATABASE_URL=postgresql://... (hardcoded temporarily)
```

## Usage Flow

1. **Setup Phase**:
   - Deploy to Railway
   - Create PostgreSQL tables
   - Configure environment variables

2. **Document Upload**:
   - Upload job descriptions, company info as .txt files
   - System chunks and indexes in Pinecone

3. **Golden Creation**:
   - Create Excel with test questions and ideal answers
   - Upload via UI or API

4. **Chat Usage**:
   - Ask questions about uploaded documents
   - Get accurate, grounded responses

5. **Evaluation**:
   - Run evaluation to test system performance
   - View metrics: accuracy and tone scores
   - Identify areas for improvement

## Current Performance

Latest evaluation results:
- **Accuracy**: 84.6% (11/13 correct)
- **Tone**: 100% (13/13 empathetic)

The system successfully:
- ✅ Answers questions accurately from uploaded documents
- ✅ Refuses to hallucinate when information isn't available
- ✅ Maintains supportive, coach-like tone
- ✅ Identifies behavioral questions requiring personal experience

## Architecture Decisions

1. **Separate Storage**: Documents in Pinecone (vector search), Goldens in PostgreSQL (structured data)
2. **Dual LLM Approach**: GPT-3.5 for answering (consistent), Gemini for judging (free tier)
3. **Fixed Embeddings**: 1024 dimensions to match Pinecone requirements
4. **Chunking Strategy**: 500 chars with overlap for context preservation

## Future Enhancements

1. **Synthetic Data Generation**: Auto-generate test variations from goldens
2. **Multi-document Support**: Handle more complex document types
3. **Advanced Analytics**: Track performance over time
4. **Prompt Optimization**: Fine-tune for better accuracy/tone balance
5. **UI Improvements**: Add file management, real-time metrics display

## Troubleshooting

Common issues:
- **"No information found"**: Document not properly indexed or metadata missing
- **PostgreSQL connection**: Ensure using public URL, not internal
- **Rate limits**: Gemini free tier allows 10 requests/minute
- **Accuracy drops**: Check if prompt is too permissive with context

## License

This project is for educational and personal use.
