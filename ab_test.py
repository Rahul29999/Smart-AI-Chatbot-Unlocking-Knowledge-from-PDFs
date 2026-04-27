
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key='')  # apni key yahan

PROMPT_A = """You are an intelligent document assistant.
Answer using ONLY the context below.
If not found, say: "Not available in document."

Context: {context}
Question: {query}
Answer:"""

PROMPT_B = """You are a helpful assistant.
Use the context to answer clearly and simply.
If unsure, say: "I could not find this in the document."

Context: {context}
Question: {query}
Answer:"""

def call_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def run_ab_test(queries: List[str], context: str) -> List[Dict]:
    results = []
    print(f"\n{'Query':<40} {'Prompt A':<50} {'Prompt B'}")
    print("-" * 130)
    for query in queries:
        answer_a = call_gpt(PROMPT_A.format(context=context, query=query))
        answer_b = call_gpt(PROMPT_B.format(context=context, query=query))
        results.append({
            "query": query,
            "prompt_a": answer_a,
            "prompt_b": answer_b
        })
        print(f"{query:<40} {answer_a[:50]:<50} {answer_b[:50]}")
    return results
