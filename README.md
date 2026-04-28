# Smart GenAI PDF Chatbot: Unlocking Knowledge from Documents

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![LangChain](https://img.shields.io/badge/LangChain-1.2-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## Project Overview

This is an end-to-end production-ready AI chatbot that extracts knowledge from PDF documents and answers user queries using a Retrieval-Augmented Generation (RAG) pipeline.

The system combines FAISS-based semantic search, LangChain prompt orchestration, and GPT-3.5 to deliver accurate, context-aware answers — with a structured FastAPI backend, A/B prompt testing, and real-time monitoring.

---

## Architecture

```
PDF Input
    |
PDF Text Extraction (pdfplumber)
    |
Sentence Preprocessing (spaCy)
    |
Embedding Generation (Sentence Transformers - all-MiniLM-L6-v2)
    |
FAISS Vector Index
    |
Query -> Semantic Search
    |-- High Confidence --> PDF Answer
    |-- Low Confidence --> LangChain + GPT-3.5 Fallback
                                |
                    FastAPI Response + Monitoring Log
```

---

## Key Features

- RAG Pipeline: Retrieval-Augmented Generation — answers from PDF first, GPT fallback when confidence is low
- FAISS Vector Search: Fast semantic nearest-neighbor search on sentence embeddings
- LangChain Integration: Prompt templates and LLMChain orchestration
- FastAPI Backend: Production-ready REST API with /upload, /ask, /health, /stats endpoints
- A/B Prompt Testing: Compare two prompt strategies to measure which gives better answers
- Monitoring and Logging: Query logs, source tracking, latency tracking per request
- Evaluation Framework: Routing accuracy metrics — PDF vs GPT source evaluation
- Docker Ready: Fully containerized for deployment

---

## Project Structure

```
Smart-AI-Chatbot/
|-- Ai_Chatbot.ipynb        # Original experimental notebook
|-- chatbot.py              # Core RAG pipeline (extract, embed, search)
|-- langchain_pipeline.py   # LangChain prompt template + LLMChain
|-- app.py                  # FastAPI server (upload, ask, stats endpoints)
|-- ab_test.py              # A/B testing for prompt comparison
|-- evaluate.py             # Evaluation framework + routing accuracy
|-- monitoring.py           # Query logging + latency tracking
|-- requirements.txt        # Dependencies
|-- Dockerfile              # Container deployment
|-- README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (IndexFlatL2) |
| LLM Orchestration | LangChain + LangChain-OpenAI |
| API Framework | FastAPI + Uvicorn |
| PDF Extraction | pdfplumber |
| NLP Preprocessing | spaCy (en_core_web_sm) |
| Containerization | Docker |
| Monitoring | Custom JSON logging |

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rahul29999/Smart-AI-Chatbot-Unlocking-Knowledge-from-PDFs.git
cd Smart-AI-Chatbot-Unlocking-Knowledge-from-PDFs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set your OpenAI API key

In chatbot.py, langchain_pipeline.py, and ab_test.py:

```python
openai.api_key = 'your-api-key-here'
```

### 4. Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Run with Docker

```bash
docker build -t pdf-chatbot .
docker run -p 8000:8000 pdf-chatbot
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Check API status and PDF load status |
| POST | /upload | Upload a PDF — extracts and indexes content |
| POST | /ask | Ask a question — returns answer, source, and latency |
| GET | /stats | Query stats — total, pdf answered, gpt answered, avg latency |

### Upload PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "PDF uploaded and indexed",
  "sentences_indexed": 923
}
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is SQL?", "threshold": 0.8}'
```

Response:
```json
{
  "query": "What is SQL?",
  "answer": "SQL is a structured query language used to manage relational databases.",
  "source": "pdf",
  "latency_ms": 22.97
}
```

### Get Stats

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "total_queries": 10,
  "pdf_answered": 8,
  "gpt_answered": 2,
  "avg_latency_ms": 23.74
}
```

---

## Evaluation

Run the evaluation script to measure routing accuracy:

```bash
python evaluate.py
```

Output:
```
Query                                         Expected   Got        Result
------------------------------------------------------------------------
What is this document about?                  pdf        pdf        PASS
Explain quantum computing in detail           gpt        gpt        PASS
What are the main topics covered?             pdf        pdf        PASS
Tell me about artificial intelligence         gpt        gpt        PASS

Routing Accuracy: 100.0% (4/4)
```

---

## A/B Prompt Testing

Compare two different prompt strategies:

```bash
python ab_test.py
```

Tests Prompt A (strict document assistant) vs Prompt B (friendly tutor style) to identify which prompt yields better, more accurate answers for a given document.

---

## Monitoring

Every query is automatically logged to query_logs.json:

```json
{
  "timestamp": "2025-04-27T11:06:00",
  "query": "What is SQL?",
  "answer": "SQL is a structured query language...",
  "source": "pdf",
  "latency_ms": 22.97
}
```

---

## Key Technical Decisions

### RAG over Fine-tuning

Deliberately chose RAG instead of fine-tuning for this use case. RAG retrieves answers directly from the PDF source, making responses accurate and traceable back to the document. Fine-tuning would risk hallucination and lose source grounding — which matters a lot when the document is the source of truth.

### FAISS over ChromaDB

FAISS is lightweight, fast, and requires no additional server setup — ideal for document-scale retrieval. ChromaDB adds unnecessary infrastructure complexity for this use case.

### LangChain for Prompt Orchestration

Used LangChain's PromptTemplate and LLMChain for structured, reusable prompt management — following production best practices over raw string formatting.

---

## Future Enhancements

- Streamlit or Gradio web UI for non-technical users
- Support for multiple PDFs simultaneously
- Local LLM support (LLaMA or Mistral) to remove API cost dependency
- Context window sliding for large document understanding
- GitHub Actions CI/CD pipeline
- Cloud deployment on AWS or GCP

---

## Use Cases

- Education: students querying textbooks and study material
- Enterprise: employees querying internal policy documents
- Legal: lawyers querying case files and contracts
- Technical: engineers querying manuals and documentation

---

## Author

Rahul Kumar Sharma

- GitHub: https://github.com/Rahul29999
- LinkedIn: https://linkedin.com/in/rahul-kumar-sharma-aa0b57233
