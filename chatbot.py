
from typing import Tuple, List
import re
import numpy as np
import faiss
import pdfplumber
import spacy
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(api_key='')  # apni key yahan

_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

def preprocess_text(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_sm")
    clean = re.sub(r"\s+", " ", text)
    return [s.text.strip() for s in nlp(clean).sents if len(s.text.strip()) > 10]

def build_index(sentences: List[str]):
    embeddings = np.array(_model.encode(sentences), dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def build_prompt(query: str, context: str) -> str:
    return f"""You are an intelligent document assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say: "This information is not available in the document."

Context:
{context}

Question: {query}

Answer:"""

def gpt_fallback(query: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": build_prompt(query, context)}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def search_query(
    query: str,
    sentences: List[str],
    index,
    threshold: float = 0.8
) -> Tuple[str, str]:
    query_vec = np.array(_model.encode([query]), dtype="float32")
    distances, indices = index.search(query_vec, k=1)
    if distances[0][0] < threshold:
        return sentences[indices[0][0]], "pdf"
    context = " ".join(sentences[:20])
    return gpt_fallback(query, context), "gpt"
    return gpt_fallback(query, context), "gpt"
