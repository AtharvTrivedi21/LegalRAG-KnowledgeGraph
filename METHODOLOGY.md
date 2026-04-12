# Graph-Constrained Legal RAG for Indian Law: Methodology

**MTech Research Work | Atharv**

## Table of Contents

1. [What is BNS Mitra and What Did It Do?](#1-what-is-bns-mitra-and-what-did-it-do)
2. [What Was Built Differently](#2-what-was-built-differently)
3. [Why These Choices Were Made](#3-why-these-choices-were-made)
4. [Embeddings: Deep Dive](#4-embeddings-deep-dive)
   - [What Are Embeddings?](#41-what-are-embeddings)
   - [Old Work vs This Work](#42-old-work-bns-mitra-style-vs-this-work)
   - [How the Embeddings Were Trained](#43-how-the-embeddings-were-trained)
   - [Why Fine-Tune at All?](#44-why-fine-tune-at-all)
5. [System Architecture](#5-system-architecture)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Summary of Novelty](#7-summary-of-novelty)


## 1. What is BNS Mitra and What Did It Do?

BNS Mitra is a Legal RAG (Retrieval-Augmented Generation) chatbot for the Bharatiya Nyaya Sanhita (BNS). Its approach is straightforward:

1. **Take a PDF** of the BNS Act.
2. **Split it into chunks** using a character-based text splitter (1000 characters per chunk, 200 character overlap).
3. **Embed those chunks** using a general-purpose embedding model (`nomic-embed-text` via Ollama).
4. **Store them in FAISS** (a vector similarity search index).
5. When a user asks a question, **rephrase it** into legal language using an LLM, **retrieve the top 4 chunks** from FAISS, and **generate an answer** using LLaMA 2.

This is essentially a standard RAG pipeline applied to a single legal PDF. It works, but it has several limitations that this project addresses.


## 2. What Was Built Differently

The proposed system is a **Graph-Constrained Legal RAG**. It combines a Knowledge Graph with vector retrieval, domain-specific embeddings, and a multi-step orchestrated pipeline. Here is a side-by-side comparison:

| Aspect | BNS Mitra (Old Work) | This Work |
|--------|---------------------|----------|
| **Data sources** | Single BNS PDF | BNS + BNSS + BSA + Constitution of India + Supreme Court Judgments (2016-2025) |
| **Data representation** | Raw PDF text | Structured CSVs: sections, articles, cases, acts, citation edges |
| **Knowledge Graph** | None | Neo4j graph with Cases, Sections, Articles, Acts, and CITES/IN_ACT relationships |
| **Embedding model** | `nomic-embed-text` (general purpose) | `BAAI/bge-small-en-v1.5` fine-tuned on 10,000 Indian legal Q&A pairs |
| **Chunking strategy** | Character-based (1000 chars, 200 overlap) | Token-based using the embedding model's own tokenizer (500 tokens, 100 overlap) |
| **Retrieval depth** | Top-4 chunks, no filtering | Top-8 chunks with graph constraints + diversity guarantees |
| **Graph constraints** | None | Retrieved chunks are filtered/boosted by Knowledge Graph relationships |
| **Source diversity** | No control | Guarantees minimum 3 section chunks + 2 article chunks per query |
| **Query understanding** | Simple rephrase | Parse sections/articles, detect Act names (BNS/BNSS/BSA/Constitution), disambiguate |
| **Orchestration** | Linear chain (rephrase > retrieve > answer) | LangGraph 5-node workflow with graph enrichment and fallback logic |
| **Legal awareness** | BNS only | Knows IPC/CrPC/IEA are repealed; references only BNS/BNSS/BSA/Constitution |
| **LLM** | LLaMA 2 (7B) | LLaMA 3 (8B) |

### The Four Phases

The system is built in four distinct phases:

**Phase 1, Data Engineering:** Text is extracted from all legal PDFs (4 Acts + thousands of Supreme Court judgments). These are parsed into structured tables: sections with section numbers, articles with article numbers, cases with case IDs, and citation edges (which case cites which section/article). Output: 5 clean CSVs ready for a database.

**Phase 2, Knowledge Graph:** Those CSVs are loaded into Neo4j to form a Knowledge Graph. Nodes are Cases, Sections, Articles, and Acts. Edges represent citations (Case > Section) and containment (Section > Act). This gives the system a *structural understanding* of how Indian law is connected.

**Phase 3, Embeddings & Vector Index:** A domain-specific embedding model is fine-tuned on Indian legal Q&A data (IndicLegalQA dataset, 10K pairs). All legal text is chunked using token-based splitting aligned with the embedding model. A FAISS index is built from all chunks.

**Phase 4, RAG Workflow:** A LangGraph-orchestrated pipeline that:
1. Parses the user query for explicit section/article references and Act mentions
2. Queries Neo4j for related sections, articles, and cases (graph retrieval)
3. Rephrases the query into formal legal language
4. Searches FAISS with graph constraints to find the most relevant chunks
5. Generates a structured answer citing specific sections and cases


## 3. Why These Choices Were Made

### Why a Knowledge Graph?

BNS Mitra treats the BNS as a flat bag of text chunks. But law is inherently *structured*: sections belong to acts, cases cite sections, and different acts relate to each other. A Knowledge Graph captures these relationships.

**Concrete benefit:** When a user asks about "Section 330 of BNS," the system doesn't just search for similar text. It looks up Section 330 in Neo4j, finds all Supreme Court cases that cite it, and uses those as constraints for vector retrieval. This means the retrieved context is both *semantically relevant* (from FAISS) and *legally grounded* (from the Knowledge Graph).

### Why Multiple Acts Instead of Just BNS?

In 2023, India repealed three colonial-era laws (IPC, CrPC, IEA) and replaced them with BNS, BNSS, and BSA. A legal assistant that only knows BNS is incomplete. Criminal procedure (BNSS) and evidence law (BSA) are equally important. The Constitution is also included because fundamental rights (Articles 14, 19, 21, etc.) are frequently cited in criminal matters.

### Why Structured Data Extraction Instead of Raw PDF?

PDFs are messy. Page numbers, headers, footers, and formatting artifacts all end up in the text. When you chunk a raw PDF, you get chunks that span across sections, mix section text with page numbers, or split a section's text in the middle of a sentence.

The Phase 1 pipeline uses regex-based parsing to extract each section and article as a clean, self-contained unit with metadata (section number, act ID, full text). This means every chunk in the FAISS index has proper metadata, so the system knows exactly which Act and which section number each chunk belongs to.

### Why Token-Based Chunking Instead of Character-Based?

Embedding models don't see characters; they see tokens. A 1000-character chunk might be 150 tokens or 300 tokens depending on the text. The chunking in this system uses the embedding model's own tokenizer to create chunks of exactly 500 tokens with 100-token overlap. This ensures every chunk is optimally sized for the embedding model, leading to better quality embeddings.

### Why LangGraph Orchestration?

A simple linear pipeline (rephrase > retrieve > answer) cannot handle complex queries well. The LangGraph workflow has 5 nodes with specific responsibilities:

- **Query Parser** detects if the user mentioned specific sections or articles, and which Act they're referring to.
- **Graph Retriever** uses the parsed information to query Neo4j for related legal entities.
- **Query Rephrase** converts informal language to formal legal terminology.
- **Vector Retriever** searches FAISS with optional graph constraints and diversity guarantees.
- **Answer Generator** produces a structured answer with headings: Summary, Applicable Laws, Relevant Case Law, Recommendation.

This multi-step approach ensures that each component does one thing well, and the pipeline handles both explicit queries ("What does Section 302 BNS say?") and natural language queries ("Someone broke into my house and stole my jewelry") gracefully.

### Why Diversity Guarantees in Retrieval?

In the corpus, approximately 97% of chunks come from Supreme Court cases (thousands of judgments) and only ~3% come from sections and articles. Without diversity control, a top-8 retrieval would almost always return 8 case chunks and zero section/article chunks. But for a legal assistant, the actual statutory text is critical.

The retriever guarantees at least 3 section chunks and 2 article chunks in every result set, ensuring the LLM always has the actual law text alongside case precedents.


## 4. Embeddings: Deep Dive

### 4.1 What Are Embeddings?

An embedding is a way to represent text as a list of numbers (a vector) such that texts with similar meanings have similar vectors. When a user asks "What is the punishment for theft?", the embedding of this question should be close to the embedding of the BNS section about theft.

Think of it like placing every piece of text on a map. Texts about similar topics end up near each other. When you search, you find the nearest points on the map.

The quality of this "map" determines how well the system retrieves relevant legal text. A general-purpose map treats all text equally. A *legal-domain* map understands that "dishonest misappropriation" and "theft" are closely related, even though the words are different.

### 4.2 Old Work (BNS Mitra Style) vs This Work

| Aspect | Old Work | This Work |
|--------|----------|----------|
| **Model** | `nomic-embed-text` (137M params, general purpose) | `BAAI/bge-small-en-v1.5` fine-tuned on legal data (33M params, domain-specific) |
| **Training** | Pre-trained on general web data, no legal fine-tuning | Pre-trained on general data, then **fine-tuned on 10,000 Indian legal Q&A pairs** |
| **Embedding dimensions** | 768 | 384 |
| **How it runs** | Via Ollama (local inference server) | Loaded directly via `sentence-transformers` (Python library) |
| **Legal understanding** | Treats legal text like any other English text | Understands legal terminology, Indian law concepts, and question-to-provision mapping |
| **Chunking alignment** | Character-based chunks (not aligned with model's tokenizer) | Token-based chunks using the model's own tokenizer |
| **Index type** | FAISS with LangChain wrapper | FAISS `IndexFlatIP` (inner product on normalized vectors = cosine similarity) |

**Why BGE over Nomic?**

`nomic-embed-text` is a good general-purpose model, but it was not designed for legal text. `bge-small-en-v1.5` is smaller (33M vs 137M parameters) but was designed specifically for information retrieval tasks. It produces 384-dimensional embeddings (vs 768), which means the FAISS index is half the size and searches are faster, while achieving better retrieval quality after fine-tuning.

The smaller model size also means it can be fine-tuned on a single GPU without running out of memory, which is practical for an MTech project setting.

### 4.3 How the Embeddings Were Trained

#### The Dataset

The **IndicLegalQA Dataset** was used: a curated collection of 10,000 question-answer pairs about Indian law. Each pair consists of:
- A **question** in natural language (e.g., "What are the provisions for bail in non-bailable offences?")
- An **answer** containing the relevant legal text and explanation

This dataset is specifically about Indian law, which is what makes it valuable for this use case.

#### The Training Process

1. **Split the data:** 80% for training (8,000 pairs), 20% for evaluation (2,000 pairs). Fixed random seed (42) for reproducibility.

2. **Load the base model:** Start with `BAAI/bge-small-en-v1.5`, a pre-trained model that already understands English well but doesn't know Indian law specifically.

3. **Training objective, Multiple Negatives Ranking Loss (MNRL):**
   This is the key technique. For each training pair (question, answer):
   - The model learns that this question and this answer should have similar embeddings (they should be close on the map).
   - All *other* answers in the same batch are treated as negatives (they should be far away).
   - With a batch size of 32, each question has 1 positive match and 31 negative matches to learn from.

   This is efficient because there is no need to manually create negative examples; the batch itself provides them.

4. **Training configuration:**
   - Epochs: 2 (deliberately low to avoid overfitting. Legal text is specialized, and over-training would make the model memorize instead of generalize)
   - Batch size: 32
   - Learning rate: 1e-5 (very small, to make careful adjustments to the pre-trained weights)
   - Warmup: 10% of training steps (gradually increase learning rate to avoid destabilizing the pre-trained weights)
   - Best model selection: Based on Recall@10 on the evaluation set (pick the checkpoint that retrieves the most relevant answers in the top 10 results)

5. **Evaluation metrics** (measured on the held-out 2,000 pairs):
   - **MRR@10** (Mean Reciprocal Rank): How high does the correct answer rank? Target: > 0.50
   - **NDCG@10** (Normalized Discounted Cumulative Gain): Quality of the ranking. Target: > 0.55
   - **Recall@10**: Is the correct answer in the top 10 at all? Target: > 0.60

6. **Safety check:** After training, the fine-tuned model is compared against the unfine-tuned baseline on Recall@10. If fine-tuning made things *worse* (which can happen with too little data or too many epochs), the system automatically falls back to the base model. This ensures a degraded model is never deployed.

#### What the Training Actually Changes

The base BGE model already knows English grammar and semantics. Fine-tuning adjusts its internal weights so that:
- Legal synonyms are placed closer together (e.g., "theft" and "dishonest misappropriation of property")
- Questions about offenses are close to the sections that define those offenses
- Indian legal terminology (BNS section numbers, legal concepts like "cognizable offense," "bailable") is properly understood
- The model distinguishes between similar but legally different concepts (e.g., "theft" vs "robbery" vs "dacoity", which are distinct BNS sections)

### 4.4 Why Fine-Tune at All?

**The core problem:** General-purpose embedding models are trained on web data: Wikipedia, news articles, forums. They understand everyday English but not legal language. When a user asks "What happens if someone breaks into my house at night?", a general model might retrieve chunks about home security or news about burglaries. A legal-domain model retrieves BNS Section 330 (house-breaking by night) and Section 331 (lurking house-trespass by night).

**Why not just use a bigger general model?** Bigger models (like `nomic-embed-text` at 137M params) are better at general text but still lack legal domain knowledge. The smaller fine-tuned model (33M params) outperforms larger general models on legal retrieval because it has been specifically taught the question-to-legal-provision mapping.

**Why IndicLegalQA?** This dataset is the bridge between how people ask legal questions and how the law is written. It teaches the model the mapping between natural language queries and legal provisions in the Indian context. A dataset about US law or European law would not help because the terminology, section numbering, and legal concepts are different.

**The result:** After fine-tuning, when the system encounters a user query, the embedding model generates a vector that is naturally close to the relevant BNS/BNSS/BSA sections, not because of keyword matching, but because it understands the *legal meaning* of the query.


## 5. System Architecture

```
User Query
    |
    v
+--------------------+
|  Query Parser      |  Detect Section/Article refs, identify Act (BNS/BNSS/BSA/Constitution)
+--------+-----------+
         |
         v
+--------------------+
| Graph Retriever    |  Neo4j lookup: find sections, articles, citing cases
| (Knowledge Graph)  |  Build graph constraints (allowed IDs)
+--------+-----------+
         |
         v
+--------------------+
|  Query Rephrase    |  LLM rephrases informal query into formal legal query
|  (LLaMA 3)        |  Aware of IPC to BNS repeal
+--------+-----------+
         |
         v
+--------------------+
| Vector Retriever   |  FAISS search with fine-tuned BGE embeddings
| (FAISS + BGE)      |  Apply graph constraints + diversity guarantees
+--------+-----------+
         |
         v
+--------------------+
| Answer Generator   |  LLM generates structured answer
| (LLaMA 3)         |  Context = graph metadata + retrieved chunks
+--------+-----------+
         |
         v
Structured Answer
(Summary, Applicable Laws, Case Law, Recommendation)
```

**Knowledge Graph Schema (Neo4j):**

```
(:Case) --[:CITES]--> (:Section) --[:IN_ACT]--> (:Act)
(:Case) --[:CITES]--> (:Article) --[:IN_ACT]--> (:Act)
```

- **Nodes:** ~10,000 Cases, ~700 Sections (BNS + BNSS + BSA), ~400 Articles (Constitution), 4 Acts
- **Edges:** Citation relationships extracted from Supreme Court judgments


## 6. Evaluation Framework

A rigorous comparison framework was built that tests three systems on the same 10 test cases:

| System | Description |
|--------|-------------|
| **System 1 (Old Work)** | BNS Mitra clone: PDF chunks + nomic-embed-text + top-4 retrieval + no graph |
| **System 2 (Simple BNS)** | Fine-tuned BGE embeddings + structured chunks + top-8 retrieval, but no graph |
| **System 3 (Full Pipeline)** | Complete system: fine-tuned BGE + graph constraints + diversity + Neo4j enrichment |

**Metrics measured across four dimensions:**

1. **Accuracy**: Section precision, recall, F1 (did the system cite the correct BNS sections?), correct Act cited (BNS, not IPC)
2. **Hallucination**: IPC reference count (should be zero), fabricated section count, grounding score
3. **Speed**: Rephrase / retrieval / generation / total latency
4. **Answer Quality**: Offense category hit, keyword coverage, completeness score, key issue coverage

This three-system comparison isolates the contribution of each improvement:
- System 1 to 2 shows the impact of **better embeddings and structured data**
- System 2 to 3 shows the impact of **the Knowledge Graph and diversity retrieval**


## 7. Summary of Novelty

Key contributions over BNS Mitra:

1. **Graph-Constrained Retrieval.** First (to the best of available knowledge) application of a Knowledge Graph to constrain vector retrieval for Indian legal RAG. The graph provides structural legal knowledge that pure vector search cannot capture.

2. **Domain-Specific Fine-Tuned Embeddings.** Instead of using a generic embedding model, a smaller, faster model was fine-tuned on 10,000 Indian legal Q&A pairs, achieving better legal retrieval quality with fewer parameters.

3. **Multi-Act Coverage.** The system covers all three new criminal codes (BNS, BNSS, BSA) plus the Constitution, with awareness of the IPC/CrPC/IEA repeal, reflecting the current legal reality of India.

4. **Structured Legal Data Pipeline.** Instead of treating legal PDFs as flat text, structured sections, articles, and citation relationships are extracted, preserving the inherent structure of law.

5. **Diversity-Aware Retrieval.** Guarantees that retrieved context always includes statutory text (sections/articles) alongside case law, preventing the 97% case-chunk dominance from overwhelming the results.

6. **LangGraph Orchestration.** A multi-step workflow that handles both explicit section queries and natural language queries, with fallback logic and graph enrichment at each step.

7. **Rigorous Comparative Evaluation.** A controlled comparison framework that isolates the contribution of each improvement (embeddings, graph, diversity) using consistent test cases and multi-dimensional metrics.
