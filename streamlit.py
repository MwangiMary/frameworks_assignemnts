#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# Helpers & Constants
DEFAULT_SAMPLE_ROWS = 50000
REQUIRED_COLS = {"publish_time", "abstract", "title", "journal", "source_x"}


def to_int_series(s, fill=0):
    """Safe conversion to integer series (handles NaN / non-numeric)."""
    return pd.to_numeric(s, errors="coerce").fillna(fill).astype(int)


# Data loading & preparation
@st.cache_data(ttl=600)
def load_metadata(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a sample of the metadata CSV and do initial cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Drop rows missing essential fields
    df = df.dropna(subset=["publish_time", "abstract", "title", "journal"]).copy()

    # Convert dates and filter to relevant years
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    df["publication_year"] = df["publish_time"].dt.year.astype(int)
    df = df[df["publication_year"].between(2019, 2023)].copy()

    # Word counts (robust)
    df["abstract_word_count"] = df["abstract"].astype(str).str.split().str.len().astype(int)
    df["title_word_count"] = df["title"].astype(str).str.split().str.len().astype(int)

    # Keep only useful columns to reduce memory, but keep others for sample display
    return df.reset_index(drop=True)


# UI Rendering Functions
def render_header():
    st.markdown(
        "<h1 style='text-align:center; color:#1f77b4'>🔬 CORD-19 Data Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Interactive exploration of COVID-19 research publications. Uses a sample "
        "of the metadata CSV for fast responsiveness."
    )


def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("Filters & Controls")

    sample_rows = st.sidebar.number_input(
        "Sample rows to load (reload required)", value=DEFAULT_SAMPLE_ROWS, step=5000, min_value=1000
    )

    # Year slider
    min_year, max_year = int(df["publication_year"].min()), int(df["publication_year"].max())
    year_range = st.sidebar.slider("Publication Year Range", min_year, max_year, (min_year, max_year), 1)

    # Journal select with search-friendly selectbox
    journals = ["All"] + sorted(df["journal"].dropna().unique().tolist())
    selected_journal = st.sidebar.selectbox("Journal", journals)

    # Source select
    sources = ["All"] + sorted(df["source_x"].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox("Source", sources)

    # Abstract word count range
    min_w, max_w = int(df["abstract_word_count"].min()), int(df["abstract_word_count"].quantile(0.95))
    word_count_range = st.sidebar.slider("Abstract Word Count", min_w, int(df["abstract_word_count"].max()), (min_w, max_w), step=10)

    # Misc options
    show_wordcloud_png = st.sidebar.checkbox("Render WordCloud as PNG (faster)", True)

    return {
        "sample_rows": sample_rows,
        "year_range": year_range,
        "selected_journal": selected_journal,
        "selected_source": selected_source,
        "word_count_range": word_count_range,
        "wordcloud_png": show_wordcloud_png,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    d = df.copy()
    yr0, yr1 = filters["year_range"]
    d = d[(d["publication_year"] >= yr0) & (d["publication_year"] <= yr1)]

    if filters["selected_journal"] != "All":
        d = d[d["journal"] == filters["selected_journal"]]

    if filters["selected_source"] != "All":
        d = d[d["source_x"] == filters["selected_source"]]

    w0, w1 = filters["word_count_range"]
    d = d[(d["abstract_word_count"] >= w0) & (d["abstract_word_count"] <= w1)]

    return d


def show_top_metrics(df: pd.DataFrame, base_df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", f"{len(df):,}", delta=f"{len(df) - len(base_df):,}" if len(base_df) else None)
    with col2:
        st.metric("Avg Abstract Length", f"{df['abstract_word_count'].mean():.0f} words")
    with col3:
        st.metric("Unique Journals", f"{df['journal'].nunique():,}")
    with col4:
        st.metric("Publication Range", f"{df['publication_year'].min()} - {df['publication_year'].max()}")


# Visualization Functions
def plot_publications_by_year(df: pd.DataFrame):
    counts = df["publication_year"].value_counts().sort_index()
    fig = px.bar(
        x=counts.index.astype(int),
        y=counts.values,
        labels={"x": "Year", "y": "Number of Publications"},
        color=counts.values,
        color_continuous_scale="Blues",
        title="Publications by Year",
        height=420,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def plot_top_journals(df: pd.DataFrame, top_n: int = 10):
    counts = df["journal"].value_counts().nlargest(top_n)
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Publications", "y": "Journal"},
        color=counts.values,
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Journals",
        height=420,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def plot_distributions(df: pd.DataFrame):
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="abstract_word_count", nbins=30, title="Abstract Word Count Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="title_word_count", nbins=20, title="Title Word Count Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)


def plot_source_pie(df: pd.DataFrame):
    counts = df["source_x"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, title="Publications by Source", height=420)
    st.plotly_chart(fig, use_container_width=True)


def render_wordcloud(df: pd.DataFrame, as_png: bool = True):
    all_titles = " ".join(df["title"].astype(str).tolist())
    if not all_titles.strip():
        st.warning("No title text available for word cloud.")
        return

    wc = WordCloud(width=1200, height=600, background_color="white", max_words=150, colormap="viridis")
    wc = wc.generate(all_titles)

    if as_png:
        # Render to a Matplotlib figure and display via st.pyplot 
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        # Directly render to the Streamlit image - convert to array first
        image = wc.to_array()
        st.image(image, use_column_width=True)


def data_sample_table(df: pd.DataFrame, n_rows: int = 10):
    columns = ["title", "journal", "publication_year", "abstract_word_count", "source_x"]
    columns = [c for c in columns if c in df.columns]
    sample = df[columns].head(n_rows).copy()
    sample = sample.rename(columns={"abstract_word_count": "Abstract Words", "publication_year": "Year", "source_x": "Source"})
    st.dataframe(sample, use_container_width=True, height=320)
    with st.expander("Dataset info"):
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.write(f"Date range: {df['publication_year'].min()} - {df['publication_year'].max()}")
        st.write(f"Unique journals: {df['journal'].nunique():,}")
        st.write(f"Unique sources: {df['source_x'].nunique():,}")


def render_insights(df: pd.DataFrame):
    if df.empty:
        st.info("No data to show insights for.")
        return

    top_journal = df["journal"].value_counts().idxmax()
    top_journal_count = df["journal"].value_counts().max()
    top_year = int(df["publication_year"].value_counts().idxmax())
    top_year_count = df["publication_year"].value_counts().max()
    avg_len = df["abstract_word_count"].mean()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""
            *Publication Trends*
            - Most prolific year: *{top_year}* ({top_year_count} papers)  
            - Most prolific journal: *{top_journal}* ({top_journal_count} papers)  
            - Average abstract length: *{avg_len:.1f}* words
            """
        )

    with col2:
        # compute simple top keywords from titles
        all_titles = " ".join(df["title"].astype(str).tolist()).lower()
        tokens = [t.strip(".,!?;:()[]{}\"'") for t in all_titles.split() if t.isalpha() and len(t) > 3]
        stop_words = {"the", "this", "that", "with", "from", "have", "have", "using", "using", "paper", "study", "research"}
        tokens = [t for t in tokens if t not in stop_words]
        if tokens:
            top_kws = pd.Series(tokens).value_counts().head(6)
            st.markdown("*Top title keywords*")
            for kw, cnt in top_kws.items():
                st.write(f"- {kw} ({cnt})")
        else:
            st.write("No significant keywords found.")


# Main app runner
def main():
    st.set_page_config(page_title="CORD-19 Explorer (Derrick variant)", page_icon="🔬", layout="wide")

    render_header()

    data_path = st.sidebar.text_input("Metadata CSV path", "metadata.csv")
    # Load initial (small) sample to populate filters quickly
    sample_rows = st.sidebar.number_input("Initial preview rows (reload required)", value=20000, step=5000, min_value=1000)
    try:
        df_base = load_metadata(data_path, nrows=sample_rows)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    controls = sidebar_controls(df_base)

    # If user wants to load a larger sample, reload using cache (keyed by args)
    if controls["sample_rows"] != DEFAULT_SAMPLE_ROWS:
        df = load_metadata(data_path, nrows=int(controls["sample_rows"]))
    else:
        # use the previously loaded base sample
        df = df_base

    filtered = apply_filters(df, controls)

    show_top_metrics(filtered, df)

    # Tabs for organization
    tabs = st.tabs(["Trends", "Journals", "Keywords", "Distributions", "Data"])
    with tabs[0]:
        st.subheader("Trends")
        plot_publications_by_year(filtered)
        plot_source_pie(filtered)

    with tabs[1]:
        st.subheader("Journals")
        plot_top_journals(filtered, top_n=10)

    with tabs[2]:
        st.subheader("Keywords")
        render_wordcloud(filtered, as_png=controls["wordcloud_png"])

    with tabs[3]:
        st.subheader("Distributions")
        plot_distributions(filtered)

    with tabs[4]:
        st.subheader("Data Sample")
        data_sample_table(filtered, n_rows=10)

    st.markdown("---")
    render_insights(filtered)
    st.markdown(
        "<div style='text-align:center; color:#666; margin-top:1rem'>🔬 CORD-19 Data Explorer — Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()