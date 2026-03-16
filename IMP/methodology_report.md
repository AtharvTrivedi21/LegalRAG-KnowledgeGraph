# Paper 1: Legal Query RAG (LQ-RAG)
**Wahidur et al. — IEEE Access, February 2025**

---

## Overview

The authors build LQ-RAG, a Retrieval-Augmented Generation pipeline tailored for legal question answering. The system has two main layers — a Fine-Tuning (FT) Layer and a RAG Layer — and addresses the twin problems of hallucination and poor domain specificity that plague general-purpose LLMs when applied to law.

## Data & Knowledge Base

**Legal corpora:** Unstructured legal books and documents collected from Library Genesis (open-access portal). The corpus is split into C_sub-legal (used for embedding fine-tuning) and C_rem (used as the live RAG retrieval store).

**Synthetic dataset for embedding:** OpenAI GPT-3.5-turbo breaks C_sub-legal into chunks and auto-generates question–context pairs, forming D_synthetic. This set is split into train/eval subsets.

**Generative fine-tuning datasets:** Two datasets are combined: Legal_QA (97,500 samples, 63%) for domain knowledge and Alpaca_cleaned (52,800 samples, 34%) for instruction-following. Smaller evaluation datasets cover TruthfulQA, SQuAD_v2, BIG-Bench Hard, MMLU Law, and LegalBench tasks (Abercrombie, LRC, CTCO, CQA, Law Stack Exchange).

## Embedding LLM Fine-Tuning

Base model: **GIST Large Embedding v0** (0.33B params, 1024-dim, BERT-based) from Hugging Face. Fine-tuned via the LlamaIndex SentenceTransformers API. Loss function: **Multiple Negatives Ranking Loss (MNRL)**, which pulls embeddings of matching query–context pairs together while pushing non-matching pairs apart using dot-product scoring. Batch sizes of 8 and 10 were tried across 3–15 epochs. The resulting model is called **GIST-Law-Embed**.

## Generative LLM Fine-Tuning & Merging

Base model: **LLaMA-3-8B** (decoder-only, 32 heads, 32 layers, 4096-dim embeddings, 15T pretrain tokens). Fine-tuning uses PEFT with **LoRA** (Low-Rank Adaptation) to keep compute tractable, plus 4-bit quantization via BitsAndBytesConfig and early stopping with weight decay. Two separate LoRA adapters are trained — one on Legal_QA (M_QA) and one on Alpaca_cleaned (M_Instr). The two adapters are then **linearly merged** into the Hybrid Fine-Tuned Model (HFM). Trainer: SFTTrainer from Hugging Face TRL library. Objective: maximize log-likelihood over target tokens.

## RAG Pipeline

C_rem is ingested using parallel workers, chunked, and encoded by GIST-Law-Embed into d-dimensional vectors, stored in a **FAISS** (Facebook AI Similarity Search) index. At query time, the user query is embedded and routed through a **ReAct** (Reasoning and Action) agent that selects the appropriate query engine tool. Retrieval is hybrid — **BM25** (lexical, TF-IDF-based) plus **Dense Passage Retrieval** (DPR, dot-product similarity over FAISS vectors). The hybrid ranker combines and re-ranks results; a re-ranker further narrows the top chunks (k=15 selected as optimal). The final prompt packs system instructions, user query, and re-ranked context, and is fed to HFM to generate an initial response.

## Evaluation Agent & Feedback Loop

An evaluation agent powered by **GPT-4** assesses every response on three criteria — Answer Relevance, Context Relevance, and Groundedness — using Chain-of-Thought (CoT) reasoning. If the response passes all thresholds, it is returned as the final output. Otherwise, a prompt engineering agent rewrites the query (simplifying it while preserving intent), and the entire retrieval-generation cycle repeats, up to N times.

## Key Results

| Metric | Result |
|---|---|
| Embedding — Hit Rate improvement (post fine-tune) | +13% |
| Embedding — MRR improvement (post fine-tune) | +15% |
| GIST-Law-Embed Avg. Hit Rate @ Top-5 | 51% |
| HFM vs. LLaMA-3-8B on general tasks | +9% |
| HFM vs. LLaMA-3-8B on legal-specific tasks | +38% |
| LQ-RAG vs. Naive RAG (relevance score) | +23% |
| LQ-RAG vs. RAG + fine-tuned LLM only | +14% |
| Average query latency | ~14.6 s |

---
---

# Paper 2: BNS Mitra — RAG-Optimized LLM Legal Virtual Assistant
**Patil et al. — ICCSAI 2025 (IEEE)**

---

## Overview

BNS Mitra is a legal chatbot designed specifically for the Bharatiya Nyaya Sanhita (BNS) — India's newly enacted criminal code that replaced the Indian Penal Code in 2023. The system takes an informal description of an incident from a user, rephrases it into formal legal language, and retrieves the most applicable BNS sections. Its intended users span the full legal spectrum: police officers filing FIRs, advocates preparing case briefs, judges verifying charge sheets, law students, and ordinary citizens.

## Architecture — Three Core Components

**1. LLM (Meta LLaMA 2 7B):** Serves as the main reasoning and language engine. It performs two roles: (a) rephrasing informal user queries into formal legal terminology via a custom `rephrase_query` function, and (b) generating the final natural-language response that explains which BNS sections apply and why. The model runs locally via **Ollama** (OllamaLLM), eliminating the need for external API calls.

**2. Knowledge Base:** The official Bharatiya Nyaya Sanhita 2023 document (PDF) is the sole knowledge source. Text is extracted from the PDF and chunked into coherent segments using LangChain's **RecursiveCharacterTextSplitter**. Each chunk is encoded into a dense vector using **OllamaEmbeddings** and stored in a FAISS vector index. This pre-encoded store enables rapid nearest-neighbour retrieval at inference time.

**3. Retrieval Chain:** Built with **LangChain**. At query time, the rephrased query is converted to a vector and compared against stored document vectors using FAISS similarity search. The most semantically relevant BNS chunks are fetched and passed back to LLaMA 2 as context for response generation.

## Workflow (End to End)

1. User submits a natural-language description of an incident through a **Streamlit** web interface.
2. The `rephrase_query` function (backed by LLaMA 2) converts the informal description into legal terminology.
3. The rephrased query is encoded into an input vector via OllamaEmbeddings.
4. FAISS searches the document vector store and returns the most relevant BNS section chunks.
5. The retrieved context and rephrased query are packaged into a prompt and fed to LLaMA 2.
6. LLaMA 2 generates the final response, naming applicable BNS sections with brief legal explanations.
7. Streamlit renders the response to the user.

## Technology Stack

| Component | Tool / Model |
|---|---|
| LLM | Meta LLaMA 2 (7B), served locally via Ollama |
| Embeddings | OllamaEmbeddings (local sentence embeddings) |
| Vector store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain (retrieval chain, prompt management, text splitting) |
| UI | Streamlit |
| Prompt engineering | Custom `rephrase_query` function — no LoRA/PEFT; model used off the shelf |

## Evaluation & Results

The chatbot was evaluated on a test set of **500 queries** drawn from mock and real case summaries spanning theft, assault, rape, mischief, and other offences. Legal experts reviewed whether the recommended BNS sections matched expectations.

| Metric | Result |
|---|---|
| Overall accuracy (expert-validated) | 87% |
| Accuracy improvement of RAG over non-RAG baseline | +12% |
| Performance on simple cases (theft, assault) | High |
| Performance on complex multi-section cases | Reduced — partial matches observed |

Prompt engineering (rephrasing queries into legal language) measurably improved retrieval precision and is confirmed as a positive finding. Accuracy dropped in cases where several BNS provisions could arguably apply simultaneously, reflecting the limits of single-hop dense retrieval without a re-ranker or iterative feedback loop.

## Limitations & Future Work

The system has no recursive feedback mechanism or evaluation agent (unlike LQ-RAG). Future improvements include integration of more advanced LLMs, expansion of the knowledge base beyond BNS, attention-based explainability methods, and an interactive multi-turn consultation mode.
