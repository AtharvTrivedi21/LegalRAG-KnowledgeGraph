from dotenv import load_dotenv
load_dotenv()

import time
from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import VECTORSTORE_DIR, ACTIVE_CONFIG

print(f"[RAG] Active config: {ACTIVE_CONFIG.config_name}")
print(f"[RAG] LLM: {ACTIVE_CONFIG.llm_model_name}")
print(f"[RAG] Embeddings: {ACTIVE_CONFIG.embedding_model_name}")

# --- Embeddings + FAISS ---

embeddings = OllamaEmbeddings(model=ACTIVE_CONFIG.embedding_model_name)

faiss_index = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = faiss_index.as_retriever(search_kwargs={"k": 4})

# --- Base LLM ---

base_llm = ChatOllama(model=ACTIVE_CONFIG.llm_model_name)

# No LangSmith-specific config; just reuse the same LLM instance
rephrase_llm = base_llm
qa_llm = base_llm

# --- Prompts ---

REPHRASE_PROMPT = PromptTemplate.from_template(
    """
You are an expert in Indian criminal law and Bharatiya Nyaya Sanhita (BNS).
Rewrite the user's informal incident description into a concise, formal legal query
using appropriate legal terminology. Do not answer the question, only rewrite it.

User incident description:
{user_query}

Formal legal query:
"""
)

QA_PROMPT = PromptTemplate.from_template(
    """
You are an AI-powered legal assistant specialized in Bharatiya Nyaya Sanhita (BNS).

You are given:
1. The original user incident description.
2. A rephrased formal legal query.
3. Relevant BNS sections with their text.

Using ONLY the information in the BNS context, answer the user's question.
Always:
- Mention the most relevant BNS section numbers (if present in the context).
- Briefly explain why these sections apply.
- Use clear, simple language understandable by a layperson.
- If you are unsure or the context is insufficient, say so and suggest consulting a human lawyer.

User description:
{user_query}

Formal legal query:
{legal_query}

BNS context:
{context}

Answer:
"""
)


# --- Core functions ---


def rephrase_query(user_query: str) -> str:
    """Rephrase user query into formal legal language using LLaMA."""
    resp = rephrase_llm.invoke(
        REPHRASE_PROMPT.format(user_query=user_query)
    )
    return resp.content.strip()


def retrieve_context(legal_query: str):
    """
    Retrieve relevant BNS chunks from FAISS using the legal query.
    Returns (docs, retrieval_time_sec).
    """
    start = time.time()
    docs = retriever.invoke(legal_query)
    retrieval_time = time.time() - start
    return docs, float(retrieval_time)


def generate_answer(user_query: str, legal_query: str, docs: List):
    """
    Generate final answer using BNS context and both queries.
    Returns (answer_text, generation_time_sec).
    """
    context = "\n\n".join(
        f"[DOC {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    start = time.time()
    resp = qa_llm.invoke(
        QA_PROMPT.format(
            user_query=user_query,
            legal_query=legal_query,
            context=context,
        )
    )
    gen_time = time.time() - start
    return resp.content.strip(), float(gen_time)


def answer_query(user_query: str) -> Dict:
    """
    Full pipeline: user query -> legal rephrase -> retrieve -> answer.

    Returns:
        {
          "rephrased_query": str,
          "answer": str,
          "docs": List[Document],
          "timings": {
              "retrieval_time_sec": float,
              "generation_time_sec": float
          }
        }
    """
    legal_query = rephrase_query(user_query)
    docs, retrieval_time = retrieve_context(legal_query)
    answer, generation_time = generate_answer(user_query, legal_query, docs)

    return {
        "rephrased_query": legal_query,
        "answer": answer,
        "docs": docs,
        "timings": {
            "retrieval_time_sec": retrieval_time,
            "generation_time_sec": generation_time,
        },
    }


if __name__ == "__main__":
    demo_q = "Someone broke into my house at night and stole my gold jewelry."
    result = answer_query(demo_q)
    print("=== USER QUERY ===")
    print(demo_q)
    print("\n=== REPHRASED LEGAL QUERY ===")
    print(result["rephrased_query"])
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== TIMINGS ===")
    print(result["timings"])