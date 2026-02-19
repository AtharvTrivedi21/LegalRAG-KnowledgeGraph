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
