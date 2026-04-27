
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=''  # apni key yahan
)

education_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are an intelligent document assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say: "This information is not available in the document."

Context:
{context}

Question: {query}

Answer:"""
)

chain = LLMChain(llm=llm, prompt=education_prompt)

def langchain_answer(query: str, context: str) -> str:
    return chain.run(context=context, query=query)
