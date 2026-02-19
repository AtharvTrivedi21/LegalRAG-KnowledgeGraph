"""
Write Mermaid .mmd files for the three architecture diagrams and optionally
export them to PNG/SVG using @mermaid-js/mermaid-cli (mmdc).

Run from project root:
  python Documentation/export_architecture_images.py

If mmdc is not installed, the script prints instructions to install it and
run mmdc manually. You can also copy each .mmd content into
https://mermaid.live and export as PNG/SVG for PPTs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
IMAGES_DIR = DOC_DIR / "images"

ARCHITECTURE_OVERVIEW_MMD = """flowchart LR
  subgraph inputs [Inputs]
    CSV[judgments CSV]
    PDFs[SC PDFs]
    Acts[Act PDFs]
  end

  subgraph phase1 [Phase 1 Ingestion]
    Load[Load judgments]
    Extract[Extract acts/sections/articles]
    Edges[Build citation edges]
    Export[Export CSVs and plots]
  end

  subgraph phase2 [Phase 2 Neo4j]
    Cypher[Cypher load]
    KG[(Knowledge Graph)]
  end

  subgraph phase3 [Phase 3 Embeddings]
    Chunk[Chunk corpus]
    Finetune[Fine-tune BGE]
    FAISS[FAISS index]
  end

  subgraph phase4 [Phase 4 RAG]
    LangGraph[LangGraph workflow]
    Ollama[Ollama LLM]
  end

  CSV --> Load
  PDFs --> Load
  Load --> Export
  Acts --> Extract
  Extract --> Export
  Edges --> Export
  Export --> Cypher
  Export --> Chunk
  Cypher --> KG
  Chunk --> Finetune
  Finetune --> FAISS
  KG --> LangGraph
  FAISS --> LangGraph
  LangGraph --> Ollama
"""

LANGGRAPH_WORKFLOW_MMD = """flowchart LR
  A[query_parser] -->|"Parse refs"| B[graph_retriever]
  B -->|"Neo4j sections and cases"| C[query_rephrase]
  C -->|"Rephrase"| D[vector_retriever]
  D -->|"FAISS and constraints"| E[answer_generator]
  E -->|"Ollama answer"| F[END]
"""

KNOWLEDGE_GRAPH_MMD = """erDiagram
  Act ||--o{ Section : IN_ACT
  Act ||--o{ Article : IN_ACT
  Case }o--|| Section : CITES
  Case }o--|| Article : CITES

  Act {
    string act_id
    string act_name
    string act_type
    string source_file
  }

  Section {
    string section_id
    string section_number
    string full_text
  }

  Article {
    string article_id
    string article_number
    string full_text
  }

  Case {
    string case_id
    int year
    string judgment_text
  }
"""


def main() -> int:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        (DOC_DIR / "architecture_overview.mmd", ARCHITECTURE_OVERVIEW_MMD),
        (DOC_DIR / "langgraph_workflow.mmd", LANGGRAPH_WORKFLOW_MMD),
        (DOC_DIR / "knowledge_graph_schema.mmd", KNOWLEDGE_GRAPH_MMD),
    ]
    for path, content in files:
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")

    # Try mmdc (global) then npx (no global install)
    def run_mmdc(args: list) -> bool:
        try:
            r = subprocess.run(
                args,
                cwd=str(DOC_DIR),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    has_mmdc = run_mmdc(["mmdc", "--version"])
    mmdc_cmd = ["mmdc"]
    if not has_mmdc:
        # Try npx (Node.js); no global install needed
        if run_mmdc(["npx", "--yes", "@mermaid-js/mermaid-cli", "mmdc", "--version"]):
            mmdc_cmd = ["npx", "--yes", "@mermaid-js/mermaid-cli", "mmdc"]

    can_export = has_mmdc or mmdc_cmd != ["mmdc"]
    if can_export:
        for mmd_name, _ in files:
            mmd_path = DOC_DIR / mmd_name
            base = mmd_path.stem
            for ext in ["png", "svg"]:
                out_path = IMAGES_DIR / f"{base}.{ext}"
                cmd = mmdc_cmd + ["-i", str(mmd_path), "-o", str(out_path)]
                try:
                    subprocess.run(cmd, cwd=str(DOC_DIR), check=True, timeout=90)
                    print(f"Exported {out_path}")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"Failed to export {out_path}: {e}")
        print("Done. Images are in Documentation/images/")
    else:
        print()
        print("mermaid-cli not found. To generate PNG/SVG for PPTs:")
        print("  Option A (Node.js): from project root run:")
        print("    python Documentation/export_architecture_images.py")
        print("    (Script will try npx; ensure Node.js is installed.)")
        print("  Option B (global): npm install -g @mermaid-js/mermaid-cli")
        print("    then: mmdc -i Documentation/architecture_overview.mmd -o Documentation/images/architecture_overview.png")
        print("  Option C: open each .mmd file in https://mermaid.live and export as PNG or SVG.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
