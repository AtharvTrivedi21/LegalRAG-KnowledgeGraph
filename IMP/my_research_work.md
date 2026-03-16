# Legal Query RAG (LQ-RAG)

## Overview
LQ-RAG, a Retrieval-Augmented Generation pipeline for legal question answering. The system has two main layers: Fine-Tuning (FT) Layer and RAG Layer.

## Data & Knowledge Base
**Legal corpora:** Unstructured legal books and documents collected. The corpus is split, used for embedding fine-tuning and for RAG retrieval.

**Synthetic dataset for embedding:** OpenAI GPT-3.5-turbo breaks C_sub-legal into chunks and auto-generates question-context pairs.

**Generative fine-tuning datasets:** Two datasets are combined: Legal_QA (97,500 samples, 63%) for domain knowledge and Alpaca_cleaned (52,800 samples, 34%) for instruction-following. Smaller evaluation datasets cover TruthfulQA, SQuAD_v2, BIG-Bench Hard, MMLU Law, LegalBench tasks (Abercrombie, LRC, CTCO, CQA), and Law Stack Exchange.

## Embedding LLM Fine-Tuning
Base model: GIST Large Embedding v0 (0.33B params, 1024-dim, BERT-based) from Hugging Face. Fine-tuned via the LlamaIndex SentenceTransformers API. Loss function: Multiple Negatives Ranking Loss (MNRL), Batch sizes of 8 and 10 were tried across 3-15 epochs. The resulting model is called GIST-Law-Embed.

## Generative LLM Fine-Tuning & Merging
Base model: LLaMA-3-8B (decoder-only, 32 heads, 32 layers, 4096-dim embeddings, 15T pretrain tokens). Fine-tuning uses PEFT with LoRA (Low-Rank Adaptation) to keep compute tractable, plus 4-bit quantization via BitsAndBytesConfig and early stopping with weight decay. Two separate LoRA adapters are trained: one on Legal_QA and one on Alpaca_cleaned. The two adapters are then linearly merged into the Hybrid Fine-Tuned Model (HFM). Trainer: SFTTrainer from Hugging Face TRL library. Objective: maximize log-likelihood over target tokens.

## RAG Pipeline
Ingestion using parallel workers, chunked, and encoded by GIST-Law-Embed into d-dimensional vectors, stored in a FAISS (Facebook AI Similarity Search) index. At query time, the user query is embedded and routed through a ReAct (Reasoning and Action) agent that selects the appropriate query engine tool. Retrieval is hybrid: BM25 (lexical, TF-IDF-based) plus Dense Passage Retrieval (DPR, dot-product similarity over FAISS vectors). The hybrid ranker combines and re-ranks results; a re-ranker further narrows the top chunks. The final prompt packs system instructions, user query, and re-ranked context, and is fed to HFM to generate an initial response.

## Evaluation Agent & Feedback Loop
An evaluation agent powered by GPT-4 assesses every response on three criteria: Answer Relevance, Context Relevance, and Groundedness, using Chain-of-Thought (CoT) reasoning. If the response passes all thresholds, it is returned as the final output. Otherwise, a prompt engineering agent rewrites the query (simplification), and the entire retrieval-generation cycle repeats, up to N times.

---

# BNS Mitra RAG-Optimized LLM Legal Virtual Assistant

## Overview
BNS Mitra is a legal chatbot designed specifically for the Bharatiya Nyaya Sanhita (BNS): India's newly enacted criminal code that replaced the Indian Penal Code in 2023. The system takes an informal description of an incident from a user, rephrases it into formal legal language, and retrieves the most applicable BNS sections.

## Architecture: Three Core Components

**1. LLM (Meta LLaMA 2 7B):** Main reasoning and language engine. It performs two roles:
- Rephrasing informal user queries into formal legal terminology via a custom rephrase_query function
- Generating the final natural-language response that explains which BNS sections apply and why. The model runs locally via Ollama (OllamaLLM).

**2. Knowledge Base:** The official Bharatiya Nyaya Sanhita 2023 document (PDF) is the sole knowledge source. Text is extracted from the PDF and chunked into segments using LangChain's RecursiveCharacterTextSplitter. Each chunk is encoded into a dense vector using OllamaEmbeddings and stored in a FAISS vector index.

**3. Retrieval Chain:** Built with LangChain. At query time, the rephrased query is converted to a vector and compared against stored document vectors using FAISS similarity search. The most semantically relevant BNS chunks are fetched and passed back to LLaMA 2 as context for response generation.

## Workflow (End to End)

**Step 1:** User submits a natural-language description of an incident through a Streamlit web interface.

**Step 2:** The rephrase_query function (backed by LLaMA 2) converts the informal description into legal terminology.

**Step 3:** The rephrased query is encoded into an input vector.

**Step 4:** FAISS searches the document vector store and returns the most relevant BNS section chunks.

**Step 5:** The retrieved context and the rephrased query are packaged into a prompt and fed to LLaMA 2.

**Step 6:** LLaMA 2 generates the final response, naming the applicable BNS sections with brief legal explanations.

**Step 7:** Streamlit shows the response to the user.

## Technology Stack
- **LLM:** Meta LLaMA 2 (7B), served locally via Ollama
- **Embeddings:** OllamaEmbeddings (local sentence embeddings)
- **Vector store:** FAISS (Facebook AI Similarity Search)
- **Orchestration:** LangChain (retrieval chain, prompt management, RecursiveCharacterTextSplitter)
- **Prompt engineering:** Custom rephrase_query function: no explicit LoRA or PEFT.

## Evaluation & Results
The chatbot was evaluated on a test set of 500 queries drawn from mock and real case summaries spanning theft, assault, mischief, and other offences. Legal experts reviewed whether the recommended BNS sections matched expectations.

- **Overall accuracy:** 87%, the suggested section was fully or closely aligned with expert opinion in 87 of every 100 queries.
- **RAG vs. no-RAG:** Incorporating RAG improved BNS section recommendation accuracy by 12% over the baseline non-RAG approach.
- **Prompt engineering impact:** Rephrasing informal queries into legal language measurably improved retrieval precision; the paper confirms this as a positive finding for RQ2.
- **Failure modes:** Accuracy dropped in multi-section or context-dependent cases where several BNS provisions could arguably apply, reflecting the limits of single-hop dense retrieval without a re-ranker or feedback loop.

## Limitations & Future Work
The system has no evaluation agent. Future improvements planned include expansion of the knowledge base beyond BNS alone.

---

*My Research Work*

---

# Our Implementation: Graph-Constrained Legal RAG for Indian Criminal Law

## Motivation and Starting Point

BNS Mitra (Patil et al.) demonstrated that a RAG-based chatbot can identify applicable BNS sections from informal incident descriptions with 87% accuracy. However, its architecture has significant limitations: a single-PDF knowledge base, flat vector retrieval with no structural awareness, off-the-shelf embeddings, and no mechanism to handle multi-act queries or produce traceable citations. Our work takes BNS Mitra's core idea — RAG for Indian criminal law — and rebuilds it from the ground up with a knowledge-graph backbone, fine-tuned embeddings, graph-constrained retrieval, and structured answer generation.

---

## What We Built (Implementation Summary)

### Phase 1 — Multi-Act Statutory Preprocessing

BNS Mitra ingests a single BNS PDF using LangChain's `RecursiveCharacterTextSplitter`, producing flat text chunks with no structural metadata.

**Our approach:**
- We ingest **four** statutory sources: BNS (2023), BNSS (2023), BSA (2023), and the Constitution of India — covering criminal law, criminal procedure, evidence law, and constitutional provisions respectively.
- We built dedicated parsers (`act_parser.py`, `constitution_parser.py`) that extract the **full hierarchical structure**: Act → Part → Chapter → Section (for BNS/BNSS/BSA) and Act → Article (for the Constitution), preserving part numbers, chapter numbers, section headings, and full legislative text.
- We extract **legal definitions** (terms defined within specific sections) and **cross-references** (sections referencing other sections or articles) using regex-based extraction pipelines.
- We produce a **structured relational output**: 15+ CSV files with normalized IDs (e.g., `BNS_2023_S302`, `CONST_1950_Art14`) that preserve the legislative hierarchy rather than discarding it during chunking.

**Improvement over BNS Mitra:** Coverage expanded from 1 act to 4 acts (including the Constitution). Structural metadata (part, chapter, heading, definitions, cross-references) is preserved rather than flattened into anonymous chunks.

### Phase 1 (continued) — Case Law Corpus

BNS Mitra uses no case law at all. The knowledge base is limited to the statutory text of BNS.

**Our approach:**
- We integrate **Supreme Court judgments** (2016–2025) from PDFs, with a triage pipeline (`sc_pdf_loader.py`) that classifies PDFs as valid judgments, skips, or errors.
- We integrate the **IL-TUR dataset** (from HuggingFace: `Exploration-Lab/IL-TUR`), a curated Indian legal text corpus.
- We run **citation extraction** (`case_citations.py`) over all case texts, using act-aware regex patterns to link each case to the specific BNS sections, BNSS sections, BSA sections, or Constitutional articles it cites, producing `case_cites_section.csv` and `case_cites_article.csv` with surrounding context snippets.

**Novelty:** Case law integration with automated citation linking is entirely absent from BNS Mitra. This enables the system to ground answers not only in statute but also in judicial interpretation.

---

### Phase 2 — Neo4j Knowledge Graph

BNS Mitra stores document chunks in a flat FAISS vector index. There is no structured representation of how legal provisions relate to each other, and no way to traverse from a section to its parent chapter, related definitions, or citing cases.

**Our approach:**
We construct a **Neo4j property graph** with the following schema:

**Node types:** Act, Part, Chapter, Section, Article, Definition, Case (7 node types)

**Relationships:**
- `Act -[:HAS_PART]→ Part -[:HAS_CHAPTER]→ Chapter -[:HAS_SECTION]→ Section` (statutory hierarchy)
- `Act -[:HAS_SECTION]→ Section`, `Act -[:HAS_ARTICLE]→ Article` (direct act membership)
- `Section -[:DEFINES_TERM]→ Definition` (definition linkage)
- `Section -[:REFERENCES]→ Section | Article` (cross-references within and across acts)
- `Case -[:CITES]→ Section | Article` (case-to-provision citations, with context property)

The graph is loaded via Cypher scripts (`01_constraints_v2.cypher`, `02_load_nodes_v2.cypher`, `03_load_edges_v2.cypher`) with uniqueness constraints on all node IDs. This graph serves as the structural backbone for retrieval: given a query about "Section 302 of BNS," the system can traverse the graph to find the section text, its parent chapter and part, definitions of terms used in it, other sections that reference it, and Supreme Court cases that cite it — all before touching the vector index.

**Novelty:** A knowledge graph with legal ontology does not exist in BNS Mitra. This is a fundamentally different retrieval paradigm — structured graph traversal combined with vector similarity — rather than vector search alone.

---

### Phase 3 — Domain-Specific Embedding Fine-Tuning

BNS Mitra uses `OllamaEmbeddings` (generic, off-the-shelf sentence embeddings) to encode document chunks. No domain adaptation is performed.

**Our approach:**
- Base model: **BAAI/bge-small-en-v1.5** (a state-of-the-art bi-encoder embedding model).
- We fine-tune on the **IndicLegalQA** dataset (10K question-answer pairs from Indian legal domain) using **Multiple Negatives Ranking Loss (MNRL)**, the same loss family used by LQ-RAG for their GIST-Law-Embed model.
- Chunking: 500-token chunks with 100-token overlap (stride 400), ensuring each chunk is semantically coherent.
- The fine-tuned model is saved as `bge-legal` and used to build a FAISS `IndexFlatIP` (inner-product / cosine similarity) index over all chunks.

**Evaluation results:**
| Metric | Value |
|--------|-------|
| MRR@10 | 0.77 |
| NDCG@10 | 0.80 |
| Recall@10 | 0.88 |

**Improvement over BNS Mitra:** Domain-adapted embeddings significantly improve retrieval precision for legal queries. BNS Mitra's off-the-shelf embeddings have no legal domain specialization, which contributes to its accuracy drop on complex multi-section cases.

---

### Phase 4 — Graph-Constrained RAG Pipeline (LangGraph)

BNS Mitra's retrieval chain is a simple LangChain pipeline: rephrase → embed → FAISS search → generate. There is no graph awareness, no diversity control, and no structured output format.

**Our approach — a multi-step LangGraph workflow (V3):**

**Step 1 — Act-Aware Query Parsing (`query_parser_v3.py`):**
The query is parsed to extract explicit section/article references and determine which act they belong to. For example, "What is Section 302 of BNS?" yields `section_number=302, section_act_id=BNS_2023`. This disambiguates sections across acts (Section 302 of BNS vs. Section 302 of old IPC) — a problem BNS Mitra does not address because it only covers one act.

**Step 2 — Graph Retrieval (`neo4j_client_v3.py`):**
Neo4j is queried for the parsed sections/articles (filtered by `act_id` when available). The graph returns:
- The full text of matched sections/articles
- Parent chapter and part context
- Cases that cite the matched provisions (with citation context)
- Cross-referenced sections

These results become `graph_constraints` — a set of allowed chunk IDs and must-include IDs that guide vector retrieval.

**Step 3 — Query Rephrase (`llm_ollama.py`):**
The informal user query is rewritten into formal legal language by the LLM (Ollama, LLaMA 3 8B), similar to BNS Mitra's `rephrase_query` function but operating within a stateful LangGraph workflow rather than as an isolated function call.

**Step 4 — Graph-Constrained Vector Retrieval (`vector_retriever_v3.py`):**
This is the core novelty in the retrieval layer. Three strategies are employed:

1. **Constrained retrieval:** FAISS retrieves `top_k × constrained_multiplier` (8 × 3 = 24) candidates, then filters to only those whose chunk IDs appear in the graph constraint set. Must-include chunks (directly matched sections/articles) are guaranteed to appear in the results.
2. **Diversity retrieval:** When no graph constraints exist, FAISS over-retrieves `top_k × diversity_multiplier` (8 × 4 = 32) candidates and applies diversity logic: at least `min_sections_per_query` (2) section chunks and `min_articles_per_query` (2) article chunks are selected, with remaining slots filled by score.
3. **Fallback:** If constrained retrieval yields fewer than `top_k` results, unconstrained retrieval fills the gap.

**Step 5 — Structured Answer Generation:**
The LLM receives a prompt containing both graph-retrieved context (exact section text, citing cases) and vector-retrieved chunks, and generates a response in a fixed structure:
- **Summary** — concise answer to the query
- **Applicable Laws / Provisions** — specific sections and articles with act names
- **Relevant Case Law** — Supreme Court or IL-TUR cases with citation context
- **Recommendation** — practical guidance

All references include traceable IDs (`section_id`, `article_id`, `case_id` with act name), enabling the user to verify every claim against the source material.

**Improvement over BNS Mitra:** The retrieval pipeline is fundamentally more sophisticated — graph structure guides vector search rather than relying on vector similarity alone. The structured output with traceable citations addresses a core weakness of BNS Mitra, which provides BNS section numbers without verifiable source links. The multi-step LangGraph workflow allows each stage to be independently tested and improved, unlike BNS Mitra's monolithic LangChain chain.

---

## Summary: Improvements over BNS Mitra

| Dimension | BNS Mitra | Our System |
|-----------|-----------|------------|
| **Knowledge base** | Single PDF (BNS only) | 4 Acts + Constitution + SC cases + IL-TUR |
| **Data representation** | Flat text chunks | Hierarchical graph (Act→Part→Chapter→Section) with definitions and cross-references |
| **Knowledge graph** | None | Neo4j with 7 node types and 8 relationship types |
| **Embeddings** | Off-the-shelf OllamaEmbeddings | BGE fine-tuned on IndicLegalQA (MRR@10 = 0.77) |
| **Retrieval** | Single-hop FAISS dense search | Graph-constrained hybrid retrieval with diversity controls |
| **Act disambiguation** | Not needed (single act) | Act-aware query parsing across BNS/BNSS/BSA/Constitution |
| **Case law** | Not included | SC judgments + IL-TUR with automated citation linking |
| **Query processing** | LangChain `rephrase_query` | LangGraph multi-step: parse → graph lookup → rephrase → constrained retrieve → generate |
| **Answer format** | Free-form LLM output | Structured: Summary + Laws + Case law + Recommendation with traceable citations |
| **Cross-references** | None | Section-to-section and section-to-article references extracted and stored in graph |
| **Definitions** | None | Legal term definitions extracted, linked to defining sections |
| **Orchestration** | LangChain retrieval chain | LangGraph stateful workflow (V3) with explicit state transitions |
| **LLM** | LLaMA 2 7B (via Ollama) | LLaMA 3 8B (via Ollama) — newer, more capable base model |

---

## Novel Contributions (Not Present in BNS Mitra)

1. **Legal Knowledge Graph with Ontology:** A Neo4j graph encoding the full structural hierarchy of Indian criminal law statutes, with typed relationships for containment, definitions, cross-references, and case citations. This enables multi-hop reasoning (e.g., "find all cases citing sections in Chapter X of BNS") that is impossible with flat vector retrieval.

2. **Graph-Constrained Vector Retrieval:** A hybrid retrieval strategy where the knowledge graph first identifies structurally relevant provisions, and these become hard constraints on FAISS vector search results. This ensures retrieval respects legal structure rather than relying solely on semantic similarity.

3. **Multi-Act Corpus with Structural Preservation:** Expanding from a single act to four interconnected statutes plus the Constitution, with full hierarchy extraction (Parts, Chapters, Sections, Articles, Definitions). This enables the system to answer queries that span multiple acts (e.g., "What BNS offence applies and what BNSS procedure governs the investigation?").

4. **Case Law Integration with Citation Extraction:** Automated extraction of section/article citations from Supreme Court judgment texts, stored as `CITES` edges in the knowledge graph. This grounds legal answers in judicial interpretation, not just statutory text.

5. **Act-Aware Query Disambiguation:** A parsing layer that identifies which act a queried section belongs to, resolving ambiguity in India's legal transition from old codes (IPC/CrPC/IEA) to new codes (BNS/BNSS/BSA).

6. **Domain-Specific Embedding Fine-Tuning on Indian Legal Data:** Fine-tuning BGE on IndicLegalQA produces embeddings that understand Indian legal terminology, improving retrieval over generic embeddings.

7. **Structured, Citation-Backed Answer Generation:** A fixed output schema (Summary, Applicable Laws, Case Law, Recommendation) with traceable source IDs, enabling legal professionals to verify every claim — a requirement for any system deployed in real legal practice.

8. **LangGraph Stateful Orchestration:** Moving from a linear LangChain retrieval chain to a LangGraph workflow with explicit state management, conditional branching, and independently testable stages.
