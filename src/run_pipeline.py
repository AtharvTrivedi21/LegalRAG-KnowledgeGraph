"""
Phase-1 Legal RAG Data Pipeline - Entry Point.
Orchestrates: load judgments, extract PDFs, build edges, export CSVs and plots.
"""
import sys
from pathlib import Path

# Add project root for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.edges import build_edges_df
from src.export import export_and_plot
from src.judgments import load_and_prepare_judgments
from src.pdf_extractor import build_acts_sections_articles


def main() -> None:
    """Run full Phase-1 pipeline."""
    base_path = config.BASE_PATH
    output_path = config.OUTPUT_PATH
    sc_extracted_dir = getattr(config, "SC_EXTRACTED_DIR", "SC_Judgements-16-25")
    judgments_source = getattr(config, "JUDGMENTS_SOURCE", "pdf")
    pdf_files = config.PDF_FILES
    act_names = config.ACT_NAMES

    # Resolve paths relative to project root
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print("Phase-1 Legal RAG Data Pipeline")
    print("-" * 40)
    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    print()

    # 1. Load judgments
    print("Loading judgments...")
    data_limit = getattr(config, "DATA_LIMIT", None)
    cases_df, schema_info = load_and_prepare_judgments(
        base_path,
        sc_extracted_dir,
        limit=data_limit,
        source=judgments_source,
    )
    print(f"  Loaded {len(cases_df)} cases")
    print(f"  Schema: {schema_info.get('shape', schema_info)}")
    stats = cases_df.groupby("year").size()
    if len(stats) > 0:
        print(f"  Years: {min(stats.index)} - {max(stats.index)}")
    print()

    # 2. Extract PDFs (acts, sections, articles)
    print("Extracting from PDFs...")
    acts_df, sections_df, articles_df = build_acts_sections_articles(
        base_path, pdf_files, act_names
    )
    print(f"  Acts: {len(acts_df)}")
    print(f"  Sections: {len(sections_df)}")
    print(f"  Articles: {len(articles_df)}")
    print()

    # 3. Build citation edges
    print("Building citation edges...")
    edges_df = build_edges_df(cases_df, articles_df, sections_df)
    print(f"  Edges: {len(edges_df)}")
    print()

    # 4. Export CSVs and plots
    print("Exporting CSVs and plots...")
    export_and_plot(
        output_path,
        cases_df,
        sections_df,
        articles_df,
        acts_df,
        edges_df,
    )
    print(f"  Output written to: {output_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
