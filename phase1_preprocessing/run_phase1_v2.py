"""
Phase 1 v2 runner: full pipeline A→F.
A: Statutory PDF preprocessing  B: Structure + enrichment  C: IL-TUR  D: SC PDFs  E: Citations  F: Validation
Run from project root: python -m phase1_preprocessing.run_phase1_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from phase1_preprocessing.pdf_text import run_statute_preprocessing
from phase1_preprocessing.structure_export import run_structure_parse_and_export
from phase1_preprocessing.iltur_download import run_iltur_normalize
from phase1_preprocessing.sc_pdf_loader import run_sc_pdf_normalize
from phase1_preprocessing.case_citations import run_case_citation_extraction
from phase1_preprocessing.validation import run_validation


def main() -> None:
    base_path = config.BASE_PATH
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    output_dir = config.PHASE1_V2_OUTPUT
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    raw_dir = output_dir / "raw_pdf_text"
    pdf_config = config.PDF_FILES

    print("[Phase1 v2] Step A: Statutory PDF preprocessing")
    print(f"  Base path: {base_path}")
    print(f"  Output:   {output_dir}")
    results = run_statute_preprocessing(base_path, pdf_config, output_dir)
    for act_id, pages in results.items():
        non_toc = sum(1 for p in pages if not p.get("is_toc") and (p.get("text") or "").strip())
        print(f"  {act_id}: {len(pages)} pages ({non_toc} non-TOC with text)")

    print("\n[Phase1 v2] Step B: Structure parse and export (+ enrichment)")
    summary = run_structure_parse_and_export(raw_dir, output_dir)
    print(f"  Acts: {summary['acts']}, Parts: {summary['parts']}, Chapters: {summary['chapters']}")
    print(f"  Sections: {summary['sections']}, Articles: {summary['articles']}")
    print(f"  Definitions: {summary.get('definitions', 0)}, Section refs: {summary.get('section_refs', 0)}, Act index: {summary.get('act_index_rows', 0)}")
    if summary["warnings"]:
        for w in summary["warnings"]:
            print(f"  WARN: {w}")

    print("\n[Phase1 v2] Step C: IL-TUR normalize")
    n_iltur = run_iltur_normalize(output_dir, base_path, allow_legacy_csv=True)
    print(f"  cases_iltur.csv: {n_iltur} rows")

    print("\n[Phase1 v2] Step D: SC judgments normalize")
    sc_dir = base_path / getattr(config, "SC_EXTRACTED_DIR", "SC_Judgements-16-25")
    if sc_dir.exists():
        cases_sc, _ = run_sc_pdf_normalize(sc_dir, output_dir, limit=getattr(config, "DATA_LIMIT", None))
        print(f"  cases_sc.csv: {len(cases_sc)} rows")
    else:
        (output_dir / "cases_sc.csv").write_text("case_id,year,full_text,source_file,source,text_extractable\n", encoding="utf-8")
        print(f"  Skip: {sc_dir} not found (empty cases_sc.csv)")

    print("\n[Phase1 v2] Step E: Case citation extraction")
    sections_csv = output_dir / "sections.csv"
    articles_csv = output_dir / "articles.csv"
    cases_sc_csv = output_dir / "cases_sc.csv"
    cases_iltur_csv = output_dir / "cases_iltur.csv" if (output_dir / "cases_iltur.csv").exists() else None
    cite_stats = run_case_citation_extraction(cases_sc_csv, cases_iltur_csv, sections_csv, articles_csv, output_dir)
    print(f"  Resolved section: {cite_stats['resolved_section']}, article: {cite_stats['resolved_article']}, unresolved: {cite_stats['unresolved']}")

    print("\n[Phase1 v2] Step F: Validation")
    val = run_validation(output_dir)
    print(f"  Report: {output_dir / 'validation_report.md'}")
    print(f"  Passed: {val['passed']}, Errors: {len(val['errors'])}, Warnings: {len(val['warnings'])}")
    print("  Done.")


if __name__ == "__main__":
    main()
