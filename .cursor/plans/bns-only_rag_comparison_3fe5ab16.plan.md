---
name: BNS-Only RAG Comparison
overview: "Build a BNS-only RAG comparison framework with three systems: (1) Old-Work baseline (PDF + nomic-embed-text + FAISS), (2) Simple-BNS (v2 structured sections + BGE embeddings + FAISS), and (3) Full-Pipeline-BNS (Neo4j + FAISS + LangGraph, filtered to BNS). Run all three on the same test cases with a new comprehensive evaluation framework and produce a comparison CSV."
todos:
  - id: phase-0-old-work
    content: "Make Old-Work runnable: copy BNS PDF to Old-Work/data/, rebuild FAISS index, standardize LLM to llama3:8b"
    status: completed
  - id: phase-1-bns-faiss
    content: Build BNS-only FAISS index from v2 sections.csv (filter to BNS_2023) with BGE embeddings
    status: completed
  - id: phase-2-adapters
    content: Create bns_comparison/ directory with three system adapters (old_work, simple_bns, full_pipeline_bns) sharing a common interface
    status: completed
  - id: phase-3-eval
    content: "Build new evaluation framework: test_cases.py with gold BNS sections, metrics.py with accuracy/hallucination/speed/quality metrics"
    status: completed
  - id: phase-4-runner
    content: Create run_comparison.py runner, execute comparison across all 3 systems, produce CSV results
    status: completed
  - id: phase-5-commit
    content: Commit and push all changes
    status: completed
isProject: false
---

