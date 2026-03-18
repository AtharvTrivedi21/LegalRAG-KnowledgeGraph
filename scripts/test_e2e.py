"""
End-to-end test of the Phase 4 V3 RAG pipeline.
Tests the full workflow: query parsing -> graph constraints -> FAISS retrieval -> Ollama LLM.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Loading pipeline (this may take 30-60s for FAISS index)...")
t0 = time.time()

from phase4_rag.langgraph_workflow_v3 import build_app

app = build_app()
print(f"Pipeline loaded in {time.time() - t0:.1f}s")

TEST_QUERIES = [
    "What is the punishment for murder under BNS?",
    "What are the fundamental rights under the Constitution of India?",
    "What is the definition of 'act' in BNS?",
]

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}: {query}")
    print('='*60)
    t1 = time.time()
    try:
        result = app.invoke({"query": query})
        elapsed = time.time() - t1

        answer = result.get("answer", "")
        chunks = result.get("retrieved_chunks", [])
        act_class = result.get("act_classification", "")
        graph_ids = result.get("graph_constraint_ids", [])

        print(f"Act classification: {act_class}")
        print(f"Graph constraint IDs ({len(graph_ids)}): {graph_ids[:5]}")
        print(f"Retrieved chunks: {len(chunks)}")
        print(f"Answer ({elapsed:.1f}s):")
        print(answer[:800] if answer else "(empty)")

        # Assertions
        if not answer:
            print("WARN: Empty answer")
        elif len(answer) < 50:
            print("WARN: Very short answer")
        else:
            print("OK: Answer looks good")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n=== E2E TEST COMPLETE ===")
