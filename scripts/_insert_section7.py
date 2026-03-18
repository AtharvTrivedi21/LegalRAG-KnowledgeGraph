"""Helper: insert section 7 into test_system.py before FINAL SUMMARY."""
from pathlib import Path

section7 = '''
# ----------------------------------------------------------------
# 7. NATURAL LANGUAGE QUERY - citations and BNS focus
# ----------------------------------------------------------------
section("7. Natural Language Query (Citation Fix Verification)")

try:
    from phase4_rag.langgraph_workflow_v3 import build_app as _build_app2
    print("  Building LangGraph app...")
    _app2 = _build_app2()

    nl_query = "Someone broke into my home and stole my property, also broke my windows and door."
    print(f"  Query: {nl_query}")
    t0 = time.time()
    nl_state = _app2.invoke({"user_query": nl_query})
    elapsed = time.time() - t0

    nl_answer  = nl_state.get("answer", "")
    nl_chunks  = nl_state.get("retrieved_chunks", [])
    nl_g_meta  = nl_state.get("graph_metadata") or {}
    nl_grouped = nl_state.get("grouped_sources") or {}
    nl_secs    = nl_g_meta.get("sections", [])

    sec_chunk_ids = list((nl_grouped.get("section") or {}).keys())
    check("Section chunks retrieved (natural language)",
          len(sec_chunk_ids) > 0,
          f"section source_ids: {sec_chunk_ids[:5]}")

    check("graph_metadata.sections populated (enrichment)",
          len(nl_secs) > 0,
          f"{len(nl_secs)} sections: {[s.get('section_id') for s in nl_secs[:3]]}")

    act_ids_found = list({s.get("act_id") for s in nl_secs if s.get("act_id")})
    check("BNS_2023 act cited",
          "BNS_2023" in act_ids_found,
          f"act_ids in metadata: {act_ids_found}")

    ipc_in_answer = "ipc" in nl_answer.lower() or "indian penal code" in nl_answer.lower()
    check("Answer does NOT reference IPC",
          not ipc_in_answer,
          "IPC found in answer" if ipc_in_answer else "clean")

    bns_in_answer = "bns" in nl_answer.lower() or "bharatiya nyaya" in nl_answer.lower()
    check("Answer references BNS",
          bns_in_answer,
          "BNS found" if bns_in_answer else "BNS not found in answer")

    check("Answer generated (natural language)", bool(nl_answer and len(nl_answer) > 50), f"{elapsed:.1f}s")

    print(f"\\n  --- Natural language answer preview ({elapsed:.1f}s) ---")
    for line in nl_answer[:700].splitlines():
        print(f"  {line}")

except Exception as e:
    import traceback; traceback.print_exc()
    check("Natural language pipeline (exception)", False, str(e))

'''

target = Path("scripts/test_system.py")
with open(target, "rb") as f:
    raw = f.read()

# Find the position of the blank line before the dashes comment before FINAL SUMMARY
marker = b"# FINAL SUMMARY"
idx = raw.rfind(marker)
blank_line_pos = raw.rfind(b"\r\n\r\n", 0, idx)
if blank_line_pos == -1:
    blank_line_pos = raw.rfind(b"\n\n", 0, idx)
    insert_pos = blank_line_pos + 2
else:
    insert_pos = blank_line_pos + 4

new_raw = raw[:insert_pos] + section7.encode("utf-8") + raw[insert_pos:]
with open(target, "wb") as f:
    f.write(new_raw)
print(f"Done. Inserted {len(section7)} chars at position {insert_pos}. New size: {len(new_raw)}")
