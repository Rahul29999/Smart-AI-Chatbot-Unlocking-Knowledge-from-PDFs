# 📚 Smart GenAI PDF Chatbot: Unlocking Knowledge from Documents

##  Project Overview

This AI chatbot leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to extract knowledge from **PDF documents** and answer user queries in natural language. It uses **FAISS (Facebook AI Similarity Search)** for fast embedding-based retrieval and integrates **Generative AI (GPT-3.5)** fallback when confidence is low — forming a **hybrid RAG system**.

---

###  Objectives

* Build a **retrieval-augmented GenAI chatbot** that answers queries using content from PDF files.
* Provide a **fallback response using GPT (or LLaMA)** when the PDF does not contain a high-confidence answer.

---

##  Key Features

* **PDF Text Extraction:** Uses `pdfplumber` to extract text from multi-page PDFs.
* **Text Preprocessing:** Applies sentence tokenization with `spaCy` for clean sentence-level retrieval.
* **Semantic Search Engine:** Encodes sentences using `SentenceTransformers` and indexes them using **FAISS**.
* **Generative AI Fallback:** Uses **OpenAI GPT-3.5** to generate answers if query confidence is low.
* **Interactive CLI Interface:** Allows users to chat in real-time.

---

##  How It Works

1. **PDF Content Extraction** – Parses and extracts raw text using `pdfplumber`.
2. **Sentence Splitting & Embedding** – Breaks text into sentences and encodes them using `all-MiniLM-L6-v2` from HuggingFace.
3. **Semantic Indexing** – Indexes sentence embeddings with FAISS for fast nearest-neighbor search.
4. **Query Matching** – Compares user query with indexed sentences. If confidence is low, a fallback LLM like **GPT** is triggered.
5. **Response Generation** – Returns either the matched sentence or a GPT-generated response based on context.

---

## ⚙ Installation

```bash
pip install pdfplumber faiss-cpu sentence-transformers openai spacy
python -m spacy download en_core_web_sm
```

---

##  Code Snippets

### 📄 PDF Text Extraction

```python
import pdfplumber

def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        return ' '.join(page.extract_text() or '' for page in pdf.pages)
```

###  Text Preprocessing

```python
import re, spacy

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    clean = re.sub(r'\s+', ' ', text)
    return [sent.text.strip() for sent in nlp(clean).sents]
```

###  Sentence Embedding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True)
```

###  FAISS Index Creation

```python
import faiss, numpy as np

embeddings_np = np.array([e.numpy() for e in embeddings])
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
```

###  GPT Fallback Logic

```python
import openai

openai.api_key = "your-openai-api-key"

def gpt_fallback(query, context):
    prompt = f"Answer this based on the document:\n\n{context}\n\nQuestion: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()
```

### Query Handler

```python
def search_query(query, sentences, index, model, embeddings, threshold=0.8):
    query_vector = model.encode([query], convert_to_tensor=True).numpy()
    distances, indices = index.search(query_vector, k=1)
    if distances[0][0] < threshold:
        return sentences[indices[0][0]], False
    else:
        context = " ".join(sentences[:20])
        return gpt_fallback(query, context), True
```

###  Chat Loop

```python
def chatbot(file_path):
    text = extract_pdf_text(file_path)
    sentences = preprocess_text(text)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    index = build_faiss_index(np.array([e.numpy() for e in sentence_embeddings]))
    
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        answer, is_gen = search_query(query, sentences, index, model, sentence_embeddings)
        print(f"{'GPT' if is_gen else 'PDF'}: {answer}\n")
```

---

##  Example Queries

| User Query                                 | Response Source     |
| ------------------------------------------ | ------------------- |
| "What is the Term End Examination policy?" |  Matched from PDF |
| "Tell me about Artificial Intelligence."   |  GPT-3.5 fallback |

---

##  Usage Instructions

1. Upload a PDF document (`your_pdf_file.pdf`) in the working directory.
2. Replace your OpenAI API key in the script.
3. Run the script in your terminal or Google Colab.
4. Ask natural questions — the bot will answer from PDF or fallback to GPT if needed.

---

##  Potential Enhancements

* Add **Streamlit or Gradio UI** for a web-based chatbot.
* Support **multiple PDFs** for document-level Q\&A.
* Use **local LLMs like LLaMA or Mistral** instead of GPT-3.5 for open-source deployment.
* Add **context window sliding** for large document understanding.

---

##  Conclusion

This project demonstrates a **fully functional Generative AI-powered document chatbot**, combining **NLP**, **semantic search**, and **LLM generation**. It's ideal for use cases like:

*  Education & Online Courses
*  Customer Support over PDFs
*  Knowledge Retrieval from Manuals, Policies, and Docs



