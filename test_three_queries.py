"""
Run the 3 example queries through the Phase 4 v2 LangGraph workflow and print responses.

Run from the project venv only:
  venv\\Scripts\\activate   # Windows
  python test_three_queries.py
"""
import os
import sys

# Unbuffer stdout so we see progress when run in background/terminal
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# Ensure project root and env
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isfile(".env"):
    print("No .env found; using defaults.")

from phase4_rag.langgraph_workflow_v2 import build_app

QUERIES = [
    "Explain Article 14 of the Constitution",
    "What does Section 302 of BNS say?",
    "Someone entered my home and stole my property",
]

def main():
    app = build_app()
    for i, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 70, flush=True)
        print(f"QUERY {i}: {query}", flush=True)
        print("=" * 70, flush=True)
        try:
            state = app.invoke({"user_query": query})
            legal = state.get("legal_query") or ""
            if legal and legal.strip() != query.strip():
                print(f"\nFormal legal query: {legal[:200]}{'...' if len(legal) > 200 else ''}", flush=True)
            answer = state.get("answer") or "(no answer)"
            print(f"\nAnswer:\n{answer}", flush=True)
            # Retrieval mix
            chunks = state.get("retrieved_chunks") or []
            by_type = {}
            for c in chunks:
                t = c.get("source_type", "?")
                by_type[t] = by_type.get(t, 0) + 1
            print(f"\nRetrieved: {by_type}", flush=True)
            err = state.get("vector_error")
            if err:
                print(f"Vector error: {err}", flush=True)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 70, flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
