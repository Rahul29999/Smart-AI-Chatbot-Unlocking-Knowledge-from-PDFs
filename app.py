
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import tempfile, os, time
from chatbot import extract_pdf_text, preprocess_text, build_index, search_query
from monitoring import log_query, get_stats

app = FastAPI(title="Smart PDF Chatbot API")

_sentences = []
_index = None

@app.get("/health")
def health():
    return {"status": "ok", "pdf_loaded": len(_sentences) > 0}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global _sentences, _index
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    text = extract_pdf_text(tmp_path)
    _sentences = preprocess_text(text)
    _index, _ = build_index(_sentences)
    os.unlink(tmp_path)
    return {
        "message": "PDF uploaded and indexed",
        "sentences_indexed": len(_sentences)
    }

class QueryRequest(BaseModel):
    query: str
    threshold: Optional[float] = 0.8

@app.post("/ask")
def ask(request: QueryRequest):
    if not _sentences:
        return {"error": "Upload a PDF first"}
    start = time.time()
    answer, source = search_query(
        request.query, _sentences, _index, request.threshold
    )
    latency = round((time.time() - start) * 1000, 2)
    log_query(request.query, answer, source, latency)
    return {
        "query": request.query,
        "answer": answer,
        "source": source,
        "latency_ms": latency
    }

@app.get("/stats")
def stats():
    return get_stats()
