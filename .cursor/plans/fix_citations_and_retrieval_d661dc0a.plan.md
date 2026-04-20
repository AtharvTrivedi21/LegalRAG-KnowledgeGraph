---
name: Fix citations and retrieval
overview: "The empty citations and IPC-focused answers stem from 5 compounding bugs: FAISS was built from v1 CSVs (wrong IDs, garbled text), diversity settings are too low, LLM prompts don't mention BNS/BNSS/BSA, and there's no graph enrichment for natural language queries. This plan fixes all of them."
todos:
  - id: rebuild-faiss
    content: "Phase 1: Update config.py to use v2 CSVs, add act_id to chunk metadata, rebuild FAISS index, verify IDs match Neo4j"
    status: completed
  - id: fix-diversity
    content: "Phase 2: Increase diversity_multiplier to 25, increase min_sections to 3"
    status: completed
  - id: fix-prompts
    content: "Phase 3: Fix REPHRASE_PROMPT (IPC->BNS), system prompt (corpus description), context block (add act_id)"
    status: completed
  - id: graph-enrichment
    content: "Phase 4: Add post-retrieval graph enrichment for natural language queries -- extract section/article IDs from FAISS chunks, look up in Neo4j, populate graph_metadata"
    status: completed
  - id: verify-e2e
    content: "Phase 5: Test with 'someone broke into my home' query, verify citations appear, BNS sections cited, no IPC references"
    status: completed
isProject: false
---

