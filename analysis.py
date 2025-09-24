#!/usr/bin/env python3

from pathlib import Path
import argparse
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# A few global plot defaults
sns.set(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def ensure_output_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def save_and_show(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def plot_publications_by_year(df: pd.DataFrame, out_dir: Path):
    """Bar chart that shows number of publications per year."""
    print("Creating publications-by-year plot...")
    # use groupby to highlight a different approach from value_counts
    counts = df.groupby("publication_year").size().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index.astype(int), counts.values, color="#4C72B0", alpha=0.8)
    ax.set_title("Publications by Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")

    # annotate bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom")

    save_and_show(fig, out_dir / "publications_by_year.png")


def plot_top_journals(df: pd.DataFrame, top_n: int, out_dir: Path):
    print("Creating top journals plot...")
    counts = df["journal"].value_counts().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=counts.values, y=counts.index, palette="magma", ax=ax)
    ax.set_xlabel("Number of Papers")
    ax.set_title(f"Top {top_n} Journals", fontweight="bold")

    # annotations
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(int(v)), va="center")

    save_and_show(fig, out_dir / "top_journals.png")


def plot_word_count_distributions(df: pd.DataFrame, out_dir: Path):
    """Two-panel distributions for abstract and title word counts using KDE + hist overlay."""
    print("Creating word count distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["abstract_word_count"], bins=25, color="#7BC96F", alpha=0.6)
    sns.kdeplot(df["abstract_word_count"], ax=axes[0], color="#2B7A0B", lw=1)
    axes[0].set_title("Abstract Word Count", fontweight="bold")
    axes[0].set_xlabel("Words")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["title_word_count"], bins=18, color="#FDB462", alpha=0.6)
    sns.kdeplot(df["title_word_count"], ax=axes[1], color="#C65A11", lw=1)
    axes[1].set_title("Title Word Count", fontweight="bold")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Count")

    save_and_show(fig, out_dir / "word_count_distributions.png")


def plot_source_pie(df: pd.DataFrame, out_dir: Path):
    """Pie chart for sources. """
    print("Creating source distribution pie chart...")
    counts = df["source_x"].value_counts()
    labels = counts.index
    sizes = counts.values

    explode = [0.1 if i == 0 else 0 for i in range(len(sizes))]  # emphasize the top source
    colors = sns.color_palette("pastel", len(sizes))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, explode=explode, colors=colors)
    ax.set_title("Publications by Source", fontweight="bold")

    save_and_show(fig, out_dir / "source_distribution.png")


def print_summary(df: pd.DataFrame, out_dir: Path):
    """Compact textual summary of the dataset used for plots."""
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    n = len(df)
    print(f"Total papers analyzed: {n}")
    if n == 0:
        print("No data to summarize.")
        return

    years = df["publication_year"].astype(int)
    print(f"Year range: {years.min()} - {years.max()}")
    print(f"Average abstract length: {df['abstract_word_count'].mean():.1f} words")
    print(f"Average title length: {df['title_word_count'].mean():.1f} words")

    top_j = df["journal"].value_counts().idxmax()
    top_j_count = df["journal"].value_counts().max()
    print(f"Most active journal: {top_j} ({top_j_count} papers)")

    top_y = df["publication_year"].value_counts().idxmax()
    top_y_count = df["publication_year"].value_counts().max()
    print(f"Year with most publications: {top_y} ({top_y_count} papers)")

    print(f"Number of unique journals: {df['journal'].nunique()}")
    print(f"Number of unique sources: {df['source_x'].nunique()}")
    print("\nGenerated images are in:", out_dir.resolve())


def load_and_prepare(path: Path, nrows: int = None) -> pd.DataFrame:
    """Load CSV, perform cleaning and create features."""
    print(f"Loading data from {path} (nrows={nrows})....")
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"Raw rows loaded: {len(df)}")

    # Keep only rows with required columns present to avoid KeyError later
    required_cols = {"publish_time", "abstract", "title", "journal", "source_x"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Convert publish_time -> datetime, then extract year
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time", "abstract", "title", "journal"])

    # Year filter (same interval as original) but using clip for robustness
    df["publication_year"] = df["publish_time"].dt.year.astype(int)
    df = df[df["publication_year"].between(2019, 2023)]

    # Word counts (robust handling of non-strings)
    df["abstract_word_count"] = df["abstract"].astype(str).str.split().str.len().fillna(0).astype(int)
    df["title_word_count"] = df["title"].astype(str).str.split().str.len().fillna(0).astype(int)

    print(f"After cleaning and filtering: {len(df)} rows remain.")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Quick CORD-19 style analysis plots (Derrick variant).")
    p.add_argument("--input", "-i", type=Path, default=Path("metadata.csv"), help="Path to metadata.csv")
    p.add_argument("--rows", "-r", type=int, default=50000, help="Number of rows to read (for sampling)")
    p.add_argument("--out", "-o", type=Path, default=Path("plots"), help="Output directory for PNGs")
    p.add_argument("--top", "-t", type=int, default=8, help="Number of top journals to show")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dir(args.out)

    df = load_and_prepare(args.input, nrows=args.rows)

    
    plot_publications_by_year(df, args.out)
    plot_top_journals(df, top_n=args.top, out_dir=args.out)
    plot_word_count_distributions(df, args.out)
    plot_source_pie(df, args.out)

    print_summary(df, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()