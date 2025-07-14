#  Smart AI Chatbot: Unlocking Knowledge from PDFs

## Project Overview
This AI chatbot utilizes **Natural Language Processing (NLP)** and **Machine Learning** to extract information from a **PDF document** and answer user queries efficiently. It employs **FAISS (Facebook AI Similarity Search)** for embedding-based retrieval and provides a fallback response when the answer is not found in the PDF.

### Objective
- Develop an AI chatbot that answers queries based on the content of a provided PDF.
- Implement a fallback response for queries not covered by the document.

## Features
- **PDF Text Extraction:** Extracts content from a PDF document.
- **Text Preprocessing:** Cleans and tokenizes text for better query matching.
- **ML-Based Search:** Uses `SentenceTransformers` for sentence embeddings, indexed with **FAISS**.
- **Fallback Mechanism:** If a query isn’t matched, a predefined fallback message is displayed.

## How It Works
1. **PDF Content Extraction** - Extracts text using `pdfplumber`.
2. **Sentence Encoding** - Converts text into vector embeddings using `SentenceTransformers`.
3. **Search Mechanism** - Matches user queries to embeddings in **FAISS**.
4. **Fallback Logic** - Uses GPT as a backup if the confidence score is low.
5. **User Interaction** - Runs interactively, providing answers from the PDF or fallback responses.

## Installation
```sh
pip install pdfplumber faiss-cpu sentence-transformers
```

## Implementation
### Extract Text from PDF
```python
import pdfplumber

def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text
```

### Preprocess Text
```python
import re
import spacy

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]
```

### Generate Sentence Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True)
```

### Create FAISS Index
```python
import faiss
import numpy as np

embeddings_np = np.array([emb.numpy() for emb in embeddings])
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
```

### Query Function
```python
def search_query(query, sentences, index, model, threshold=0.5):
    query_embedding = model.encode([query], convert_to_tensor=True).numpy()
    distances, indices = index.search(query_embedding, k=1)
    if distances[0][0] > threshold:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
    return sentences[indices[0][0]]
```

### Chatbot Interaction
```python
while True:
    user_query = input("Ask a question (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        print("Goodbye!")
        break
    response = search_query(user_query, sentences, index, model)
    print(f"Response: {response}")
```

## Example Queries
- **Input:** "What is the Term End Examination policy?"
  - **Output:** "Term End Examination Credence is 70%."
- **Input:** "Tell me about AI."
  - **Output:** "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

## Usage Instructions
1. Upload a **PDF document** in Google Colab.
2. Run the provided Python scripts step by step.
3. Start the chatbot and ask questions interactively.

## Potential Enhancements
- **GPT Integration:** Add GPT fallback for better responses.
- **Web UI:** Use **Streamlit** or **Gradio** for a user-friendly interface.
- **Multi-PDF Support:** Extend functionality to process multiple PDFs.

## Conclusion
This AI chatbot provides an **efficient** and **scalable** solution for retrieving answers from PDFs using **Machine Learning** and **Natural Language Processing (NLP)**. It can be used for **education, customer support, and research applications**.

