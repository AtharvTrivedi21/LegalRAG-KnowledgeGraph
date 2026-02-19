"""
Export clean CSVs (Neo4j-ready) and save matplotlib plots.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(output_path: Path) -> Path:
    """Create output directory if needed."""
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def export_csvs(
    output_path: Path,
    cases_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    acts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:
    """Export all DataFrames to CSVs (Neo4j-ready)."""
    ensure_output_dir(output_path)
    cases_df.to_csv(output_path / "cases.csv", index=False)
    sections_df.to_csv(output_path / "sections.csv", index=False)
    articles_df.to_csv(output_path / "articles.csv", index=False)
    acts_df.to_csv(output_path / "acts.csv", index=False)
    edges_df.to_csv(output_path / "edges.csv", index=False)


def plot_judgments_per_year(cases_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: judgments per year."""
    ensure_output_dir(output_path)
    if cases_df.empty or "year" not in cases_df.columns:
        return
    counts = cases_df.groupby("year").size()
    counts = counts[counts.index > 0]  # Exclude year=0 placeholder
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Judgments")
    ax.set_title("Judgments per Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "judgments_per_year.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_cited_sections(edges_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Horizontal bar chart: top cited sections."""
    ensure_output_dir(output_path)
    if edges_df.empty:
        return
    sec_edges = edges_df[edges_df["target_section_or_article"].str.contains("_Sec_", na=False)]
    if sec_edges.empty:
        return
    top = sec_edges["target_section_or_article"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top.values, color="darkgreen", alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Citation Count")
    ax.set_title("Top Cited Sections")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / "top_cited_sections.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_cited_articles(edges_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Horizontal bar chart: top cited articles."""
    ensure_output_dir(output_path)
    if edges_df.empty:
        return
    art_edges = edges_df[edges_df["target_section_or_article"].str.contains("_Art_", na=False)]
    if art_edges.empty:
        return
    top = art_edges["target_section_or_article"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top.values, color="darkblue", alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Citation Count")
    ax.set_title("Top Cited Articles")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / "top_cited_articles.png", dpi=150, bbox_inches="tight")
    plt.close()


def export_and_plot(
    output_path: Path,
    cases_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    acts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:
    """Export all CSVs and generate plots."""
    export_csvs(output_path, cases_df, sections_df, articles_df, acts_df, edges_df)
    plot_judgments_per_year(cases_df, output_path)
    plot_top_cited_sections(edges_df, output_path)
    plot_top_cited_articles(edges_df, output_path)
