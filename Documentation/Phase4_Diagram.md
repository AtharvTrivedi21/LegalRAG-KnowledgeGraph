# Phase 4: RAG (V3) Data Flow Diagram

```mermaid
flowchart LR
  subgraph parse [Query Parser]
    Q[User query]
    PQ[ParsedQuery]
    Q --> PQ
  end
  subgraph graphRetriever [Graph Retriever]
    PQ --> G
    G[get_sections_by_numbers with act_id]
    G --> M[graph_metadata with act_name]
  end
  subgraph vectorRetriever [Vector Retriever]
    V[retrieve_chunks]
    V --> T[top_faiss_similarity]
  end
  subgraph answerGen [Answer Generator]
    T --> Guard{top < threshold?}
    Guard -->|Yes| Msg["Insufficient relevant statutory context found"]
    Guard -->|No| Gen[LLM + Act-aware context]
    Gen --> Out[Answer + applicable_acts]
  end
  parse --> graphRetriever
  graphRetriever --> vectorRetriever
  vectorRetriever --> answerGen
```

## Optional: two-stage retrieval with cross-encoder (System S4)

The default V3 flow above passes **FAISS-ranked chunks** straight to the answer generator. The **rerank experiment** (`rerank_experiment/adapter.py`) inserts a **cross-encoder** between dense retrieval and prompt construction:

```mermaid
flowchart LR
  subgraph dense [Stage 1: bi-encoder]
    V2[retrieve_chunks FAISS]
    K2[top-K candidates]
    V2 --> K2
  end
  subgraph rerank [Stage 2: cross-encoder]
    CE[ms-marco-MiniLM cross-encoder]
    S2[sort by rerank_score]
    K2 --> CE --> S2
  end
  subgraph gen [Answer generation]
    S2 --> P[build context top-k]
    P --> LLM[LLM structured answer]
  end
```

Use this block in the thesis as the **reranker architecture** figure (see `MTech Thesis/content/figures.md` Figure 6).
