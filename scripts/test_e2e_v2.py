"""
Focused end-to-end test of the Phase 4 V3 RAG pipeline.
Tests both natural language queries and explicit section/article references.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Loading pipeline...")
t0 = time.time()
from phase4_rag.langgraph_workflow_v3 import build_app
app = build_app()
print(f"Pipeline loaded in {time.time() - t0:.1f}s\n")

# Warm up Ollama with a tiny request first
from phase4_rag.llm_ollama import ChatMessage, chat_completion, OllamaError
print("Warming up Ollama (first call may fail with CUDA init)...")
for attempt in range(3):
    try:
        r = chat_completion([ChatMessage(role="user", content="Say: OK")])
        print(f"Ollama warm-up OK: {r[:30]}")
        break
    except OllamaError as e:
        print(f"  Attempt {attempt+1} failed: {str(e)[:60]}, retrying in 3s...")
        time.sleep(3)

print()

TEST_QUERIES = [
    # Natural language -- uses FAISS fallback
    ("Natural language: murder punishment", "What is the punishment for murder under BNS?"),
    # Explicit section -- triggers graph constraints
    ("Explicit section ref", "What does Section 103 of BNS say about murder?"),
    # Constitution article
    ("Constitution article", "What does Article 21 of the Constitution say about right to life?"),
    # Definition query
    ("Definition query", "What is the definition of act in BNS section 2?"),
]

results = []
for label, query in TEST_QUERIES:
    print(f"{'='*60}")
    print(f"[{label}]")
    print(f"Query: {query}")
    t1 = time.time()
    try:
        state = app.invoke({"user_query": query})
        elapsed = time.time() - t1

        answer = state.get("answer", "")
        chunks = state.get("retrieved_chunks", [])
        graph_meta = state.get("graph_metadata") or {}
        constraints = state.get("graph_constraints")
        legal_query = state.get("legal_query", "")

        sections_found = len(graph_meta.get("sections", []))
        articles_found = len(graph_meta.get("articles", []))
        cases_found = len(graph_meta.get("cases", []))

        print(f"Rephrased query: {legal_query[:80]}")
        print(f"Graph: {sections_found} sections, {articles_found} articles, {cases_found} cases")
        print(f"Graph constraints active: {constraints is not None}")
        print(f"FAISS chunks retrieved: {len(chunks)}")
        print(f"Answer ({elapsed:.1f}s):")
        print(answer[:600] if answer else "(empty)")

        ok = bool(answer and len(answer) > 50 and "error" not in answer.lower()[:100])
        results.append((label, ok, elapsed))
        print(f"Status: {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append((label, False, 0))
    print()

print("="*60)
print("SUMMARY")
print("="*60)
for label, ok, elapsed in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label} ({elapsed:.1f}s)")

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n{passed}/{len(results)} tests passed")
