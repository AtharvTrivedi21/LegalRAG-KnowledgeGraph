import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import BNS_PDF_PATH, VECTORSTORE_DIR, ACTIVE_CONFIG


def load_bns_pdf(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"BNS PDF not found at {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def chunk_documents(docs):
    # Close to what’s usually done in legal RAG setups
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_and_save_index(chunks):
    print(f"[INGEST] Using embeddings model: {ACTIVE_CONFIG.embedding_model_name}")
    embeddings = OllamaEmbeddings(model=ACTIVE_CONFIG.embedding_model_name)

    print("[INGEST] Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(VECTORSTORE_DIR), exist_ok=True)
    db.save_local(VECTORSTORE_DIR)
    print(f"[INGEST] Saved FAISS index to: {VECTORSTORE_DIR}")


def main():
    print("[INGEST] Loading BNS PDF...")
    docs = load_bns_pdf(BNS_PDF_PATH)
    print(f"[INGEST] Loaded {len(docs)} pages from BNS PDF")

    print("[INGEST] Splitting into chunks...")
    chunks = chunk_documents(docs)
    print(f"[INGEST] Created {len(chunks)} chunks")

    build_and_save_index(chunks)
    print("[INGEST] Done.")


if __name__ == "__main__":
    main()