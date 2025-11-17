from pathlib import Path
import ntpath
from typing import Dict, List
from urllib.parse import unquote

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent


def _filename_from_path(path_str: str) -> str:
    """Return bare filename from any path-like string."""
    if not isinstance(path_str, str):
        return ""
    path_str = path_str.strip().strip('"').strip("'")
    return ntpath.basename(path_str)


def _clean_filename(name: str) -> str:
    """Human-friendly version of a PDF filename (decode %20, replace _)."""
    if not isinstance(name, str):
        return ""
    name = unquote(name)
    return name.replace("_", " ")


def _pretty_label(label: str) -> str:
    """Nicely format a fraud_type-style label for display."""
    if not isinstance(label, str):
        return ""
    return label.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    """Load mapping, keyword, semantic chunk data, and LLM summaries if present."""
    mapping_path = DATA_DIR / "fincen_fraud_mapping.csv"
    keyword_path = DATA_DIR / "fincen_keyword_locations.csv"
    chunks_path = DATA_DIR / "fincen_semantic_chunks.csv"
    summaries_path = DATA_DIR / "fincen_advisory_summaries.csv"

    dfs: Dict[str, pd.DataFrame] = {}

    # ---- Mapping (core document-level fraud counts) ----
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)

        # Normalize article_name if needed
        if "article_name" not in mapping.columns:
            for candidate in ["file", "filename", "pdf"]:
                if candidate in mapping.columns:
                    mapping["article_name"] = mapping[candidate].map(_filename_from_path)
                    break
            else:
                mapping["article_name"] = ""

        # Human title
        if "doc_title" not in mapping.columns:
            mapping["doc_title"] = mapping["article_name"].apply(_clean_filename)

        # Year
        if "doc_date" in mapping.columns:
            mapping["doc_year"] = pd.to_datetime(
                mapping["doc_date"], errors="coerce"
            ).dt.year
        else:
            mapping["doc_year"] = None

        # ---- Attach LLM summaries if available ----
        if summaries_path.exists():
            summaries = pd.read_csv(summaries_path)

            # Ensure we have article_name to join on
            if "article_name" not in summaries.columns and "pdf_filename" in summaries.columns:
                summaries["article_name"] = summaries["pdf_filename"]

            summary_cols = [
                "article_name",
                "doc_type",
                "doc_title",
                "doc_date",
                "fincen_id",
                "high_level_summary",
                "primary_fraud_types",
                "key_red_flags",
                "recommended_sar_focus",
                "why_it_matters_for_usaa",
            ]
            existing_summary_cols = [c for c in summary_cols if c in summaries.columns]

            if existing_summary_cols:
                mapping = mapping.merge(
                    summaries[existing_summary_cols].drop_duplicates(subset=["article_name"]),
                    on="article_name",
                    how="left",
                    suffixes=("", "_from_summary"),
                )

                # Prefer doc_type from summaries if mapping has only "Advisory"/blank
                if "doc_type_from_summary" in mapping.columns:
                    mapping["doc_type"] = mapping["doc_type_from_summary"].combine_first(
                        mapping.get("doc_type")
                    )
                    mapping = mapping.drop(columns=["doc_type_from_summary"])

                # Optionally fill missing title/date/ID from summaries
                for col in ["doc_title", "doc_date", "fincen_id"]:
                    from_col = f"{col}_from_summary"
                    if from_col in mapping.columns:
                        mapping[col] = mapping[col].combine_first(mapping[from_col]) \
                            if col in mapping.columns else mapping[from_col]
                        mapping = mapping.drop(columns=[from_col])

        dfs["mapping"] = mapping

    # ---- Keyword-level locations ----
    if keyword_path.exists():
        kw = pd.read_csv(keyword_path)
        if "article_name" not in kw.columns and "file" in kw.columns:
            kw["article_name"] = kw["file"].map(_filename_from_path)
        for col in ["doc_type", "doc_date", "doc_title", "fincen_id"]:
            if col not in kw.columns:
                kw[col] = ""
        kw["doc_year"] = pd.to_datetime(kw["doc_date"], errors="coerce").dt.year
        dfs["keywords"] = kw

    # ---- Chunk-level semantic labels ----
    if chunks_path.exists():
        ch = pd.read_csv(chunks_path)
        if "article_name" not in ch.columns and "file" in ch.columns:
            ch["article_name"] = ch["file"].map(_filename_from_path)
        for col in ["doc_type", "doc_date", "doc_title", "fincen_id"]:
            if col not in ch.columns:
                ch[col] = ""
        ch["doc_year"] = pd.to_datetime(ch["doc_date"], errors="coerce").dt.year
        dfs["chunks"] = ch

    return dfs



def explode_top_labels(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Transform top_labels_regex into per-fraud-type rows.

    Example:
        "money_laundering:4; structuring_smurfing:1"
    ->
        rows with fraud_type="money_laundering", count=4 etc.
    """
    if "top_labels_regex" not in mapping.columns:
        return pd.DataFrame()

    records: List[dict] = []
    for _, row in mapping.iterrows():
        base = {
            "article_name": row.get("article_name", ""),
            "fincen_id": row.get("fincen_id", ""),
            "doc_type": row.get("doc_type", ""),
            "doc_title": row.get("doc_title", ""),
            "doc_date": row.get("doc_date", ""),
            "doc_year": row.get("doc_year", None),
        }
        labels_str = str(row.get("top_labels_regex", "") or "")
        if not labels_str.strip():
            continue

        parts = [p.strip() for p in labels_str.split(";") if p.strip()]
        for part in parts:
            # Expect "fraud_type:count"
            if ":" in part:
                fraud_type, count_str = part.split(":", 1)
                fraud_type = fraud_type.strip()
                try:
                    count = int(count_str.strip())
                except ValueError:
                    count = 1
            else:
                fraud_type = part.strip()
                count = 1

            rec = dict(base)
            rec["fraud_type"] = fraud_type
            rec["count"] = count
            records.append(rec)

    return pd.DataFrame.from_records(records)


def sidebar_filters(exploded: pd.DataFrame) -> Dict[str, any]:
    """Build sidebar controls and return a dict of filter values."""
    st.sidebar.header("Filters")

    all_doc_types = sorted(exploded["doc_type"].dropna().unique())
    doc_types = st.sidebar.multiselect(
        "Document types",
        options=all_doc_types,
        default=all_doc_types or None,
    )

    all_fraud_types = sorted(exploded["fraud_type"].dropna().unique())
    fraud_types = st.sidebar.multiselect(
        "Fraud types (semantic/regex labels)",
        options=all_fraud_types,
        default=all_fraud_types or None,
    )

    year_min = int(exploded["doc_year"].min()) if exploded["doc_year"].notna().any() else 1990
    year_max = int(exploded["doc_year"].max()) if exploded["doc_year"].notna().any() else 2030

    year_range = st.sidebar.slider(
        "Document year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    max_fraud_series = st.sidebar.slider(
        "Max distinct fraud types per document (for plots)",
        min_value=1,
        max_value=20,
        value=8,
    )

    return {
        "doc_types": doc_types,
        "fraud_types": fraud_types,
        "year_range": year_range,
        "max_fraud_series": max_fraud_series,
    }


def apply_filters(exploded: pd.DataFrame, filters: Dict[str, any]) -> pd.DataFrame:
    """Filter exploded doc-level fraud rows according to sidebar."""
    df = exploded.copy()

    if filters["doc_types"]:
        df = df[df["doc_type"].isin(filters["doc_types"])]

    if filters["fraud_types"]:
        df = df[df["fraud_type"].isin(filters["fraud_types"])]

    ymin, ymax = filters["year_range"]
    df = df[(df["doc_year"] >= ymin) & (df["doc_year"] <= ymax)]

    return df


def layout_overview(exploded: pd.DataFrame, max_fraud_series: int):
    """Overview tab with high-level charts and metrics."""
    st.subheader("High-level fraud signal overview")

    if exploded.empty:
        st.info("No fraud labels found. Check that `top_labels_regex` is populated.")
        return

    # ---- Summary metrics ----
    col1, col2, col3 = st.columns(3)

    total_docs = exploded["article_name"].nunique()
    total_fraud_events = int(exploded["count"].sum())
    distinct_labels = exploded["fraud_type"].nunique()

    with col1:
        st.metric("Documents with fraud labels", f"{total_docs}")
    with col2:
        st.metric("Total fraud label counts", f"{total_fraud_events}")
    with col3:
        st.metric("Distinct fraud labels", f"{distinct_labels}")

    st.markdown("---")

    # ---- Fraud labels, top N ----
    st.markdown("### Top fraud labels (by total count)")

    label_counts = (
        exploded.groupby("fraud_type")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    top_n = label_counts.head(max_fraud_series)
    st.bar_chart(top_n, width="stretch")

    st.caption(
        "Counts reflect the sum of regex-based label hits per document, "
        "as parsed from `top_labels_regex`."
    )

    st.markdown("---")

    # ---- Timeline of fraud label intensity ----
    st.markdown("### Fraud label intensity over time")

    # For each (year, label), sum counts
    yearly = (
        exploded.groupby(["doc_year", "fraud_type"])["count"]
        .sum()
        .reset_index()
    )

    if yearly.empty:
        st.info("No year information available for timeline plot.")
        return

    pivot = yearly.pivot_table(
        index="doc_year",
        columns="fraud_type",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    # Keep only up to max_fraud_series, otherwise lines get noisy
    top_labels = label_counts.head(max_fraud_series).index.tolist()
    pivot = pivot[top_labels]

    st.line_chart(pivot, width="stretch")

    st.caption(
        "Each line shows how often that fraud label appears across FinCEN documents per year."
    )


def layout_documents_table(exploded: pd.DataFrame, mapping: pd.DataFrame):
    """
    Documents tab: show FinCEN docs, their fraud labels, and LLM summaries.
    """
    st.subheader("Documents and fraud labels")

    if mapping is None or mapping.empty:
        st.info("No mapping data to show.")
        return

    docs = mapping.copy()

    # Ensure key columns exist
    for col in ["fincen_id", "doc_type", "doc_title", "doc_date", "doc_year"]:
        if col not in docs.columns:
            docs[col] = ""

    # Add aggregated fraud labels & counts
    fraud_agg = (
        exploded.groupby("article_name")
        .agg(
            total_fraud_count=("count", "sum"),
            distinct_fraud_types=("fraud_type", "nunique"),
            fraud_labels_list=("fraud_type", lambda x: sorted(set(x))),
        )
        .reset_index()
    )

    docs = docs.merge(fraud_agg, on="article_name", how="left")

    # Show a simple table
    table_cols = [
        "article_name",
        "fincen_id",
        "doc_type",
        "doc_title",
        "doc_date",
        "doc_year",
        "total_fraud_count",
        "distinct_fraud_types",
    ]
    table_cols = [c for c in table_cols if c in docs.columns]

    st.dataframe(
        docs[table_cols].sort_values("doc_date", ascending=False),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Drill into a single document")

    article_names = docs["article_name"].dropna().unique().tolist()
    if not article_names:
        st.info("No documents available to drill into.")
        return

    selected = st.selectbox(
        "Choose a FinCEN document",
        options=article_names,
        index=0,
        format_func=_clean_filename,
    )

    row = docs.loc[docs["article_name"] == selected].iloc[0]

    st.markdown(
        f"**{row.get('fincen_id', '')} – {row.get('doc_title', _clean_filename(selected))}**"
    )
    st.write(f"Type: {row.get('doc_type', '')} | Date: {row.get('doc_date', '')}")

    st.write(
        f"Total fraud label count: `{row.get('total_fraud_count', 0)}` | "
        f"Distinct fraud labels: `{row.get('distinct_fraud_types', 0)}`"
    )

    # Show label list
    labels = row.get("fraud_labels_list", [])
    if isinstance(labels, list) and labels:
        labels_fmt = ", ".join(sorted(set(labels)))
        st.write(f"Fraud labels: `{labels_fmt}`")
    else:
        st.write("Fraud labels: *(none)*")

    # Show LLM summary fields if present
    if "high_level_summary" in docs.columns:
        hls = row.get("high_level_summary", "")
        if isinstance(hls, str) and hls.strip():
            st.markdown("**High-level summary:**")
            st.write(hls)
        else:
            st.info("No LLM summary available for this document yet.")
    else:
        st.info("No LLM summary column (`high_level_summary`) found in the data.")

    if "why_it_matters_for_usaa" in docs.columns:
        wim = row.get("why_it_matters_for_usaa", "")
        if isinstance(wim, str) and wim.strip():
            st.markdown("**Why this matters for USAA:**")
            st.write(wim)


def layout_chunk_explorer(chunks: pd.DataFrame):
    st.subheader("Semantic chunk explorer")

    if chunks is None or chunks.empty:
        st.info("No semantic chunks file (`fincen_semantic_chunks.csv`) found.")
        return

    ch = chunks.copy()

    # matched_fraud_types to list
    if "matched_fraud_types" in ch.columns:
        ch["matched_fraud_types_list"] = (
            ch["matched_fraud_types"]
            .fillna("")
            .astype(str)
            .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
        )
    else:
        ch["matched_fraud_types_list"] = [[] for _ in range(len(ch))]

    all_fraud_types = sorted(
        {ft for lst in ch["matched_fraud_types_list"] for ft in lst}
    )

    st.markdown("Filter chunks by document and fraud label.")

    # Document selection
    article_names = sorted(ch["article_name"].dropna().unique())
    article_name = st.selectbox(
        "Document", options=article_names, format_func=_clean_filename
    )

    subset = ch[ch["article_name"] == article_name].copy()

    # Fraud type selection
    ft_options = ["(any)"] + all_fraud_types
    fraud_type = st.selectbox("Fraud label", options=ft_options, index=0)

    if fraud_type != "(any)":
        subset = subset[
            subset["matched_fraud_types_list"].apply(
                lambda lst: fraud_type in lst
            )
        ]

    subset = subset.sort_values(["page_number"], ascending=True)

    st.write(
        f"Showing {len(subset)} chunks for **{_clean_filename(article_name)}**"
        + (f" with fraud type `{fraud_type}`" if fraud_type != "(any)" else "")
    )

    for _, row in subset.iterrows():
        page = row.get("page_number", "?")
        snippet = row.get("text", "")
        labels_for_chunk = row.get("matched_fraud_types_list", [])
        if isinstance(labels_for_chunk, str):
            labels_for_chunk = [
                x.strip() for x in labels_for_chunk.split(",") if x.strip()
            ]
        st.markdown(f"**Page {page}** | labels: `{', '.join(labels_for_chunk)}`")
        st.write(snippet)
        st.markdown("---")


def layout_progress_report(
    exploded: pd.DataFrame,
    mapping: pd.DataFrame,
    keywords: pd.DataFrame | None,
):
    """
    Progress Report #3 tab:
    - Auto computes key stats, top fraud labels, and trends
    - Adds keyword analytics:
        * Top 5 keywords overall
        * Top 3 keywords for each of the top fraud labels
    - Shows a narrative block you can copy into your slide/report
    """
    st.subheader("Progress Report #3 – Project Snapshot")

    if mapping is None or mapping.empty or exploded is None or exploded.empty:
        st.info("No data available to build the progress report.")
        return

    # ---------- High-level stats ----------
    num_docs = mapping["article_name"].nunique()
    by_type = mapping["doc_type"].value_counts().to_dict()
    if mapping["doc_year"].notna().any():
        year_min = int(mapping["doc_year"].min())
        year_max = int(mapping["doc_year"].max())
    else:
        year_min = None
        year_max = None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total FinCEN documents", f"{num_docs}")
    with col2:
        st.metric(
            "By type",
            ", ".join(f"{k}: {v}" for k, v in by_type.items())
        )
    with col3:
        if year_min is not None and year_max is not None:
            st.metric("Year range", f"{year_min} – {year_max}")
        else:
            st.metric("Year range", "N/A")

    st.markdown("---")

    # ---------- Top 5 fraud labels / keywords ----------
    st.markdown("### Top 5 fraud labels (regex-based)")

    if "fraud_type" not in exploded.columns or "count" not in exploded.columns:
        st.info("No exploded fraud label data available.")
        return

    fraud_hits = (
        exploded.groupby("fraud_type")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    if fraud_hits.empty:
        st.info("No fraud label counts available.")
        return

    top5_labels = fraud_hits.head(5).reset_index()
    top5_labels["fraud_type_pretty"] = top5_labels["fraud_type"].apply(_pretty_label)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.table(
            top5_labels[["fraud_type_pretty", "count"]]
            .rename(columns={"fraud_type_pretty": "Fraud label", "count": "Total hits"})
        )

    with col_right:
        chart_data = top5_labels.set_index("fraud_type_pretty")["count"]
        st.bar_chart(chart_data, width="stretch")

    st.caption(
        "Fraud labels come from `top_labels_regex` and represent high-signal "
        "regex categories like money laundering, terrorist financing, ransomware, etc."
    )

    st.markdown("---")

    # ---------- Keyword analytics ----------
    st.markdown("### Keyword signals behind those fraud labels")

    if keywords is None or keywords.empty or "fraud_keyword" not in keywords.columns:
        st.info(
            "No keyword-level data found (`fincen_keyword_locations.csv`). "
            "Run the OCR/fraud mapper to generate keyword locations."
        )
    else:
        kw = keywords.copy()
        # Normalize keyword (for grouping) but keep one representative original form
        kw["kw_norm"] = kw["fraud_keyword"].astype(str).str.lower().str.strip()
        rep_map = (
            kw.groupby("kw_norm")["fraud_keyword"]
            .agg(lambda s: s.iloc[0])
            .to_dict()
        )

        # Overall Top 5 keywords
        overall_counts = (
            kw.groupby("kw_norm")
            .size()
            .sort_values(ascending=False)
            .head(5)
            .reset_index(name="Total hits")
        )
        overall_counts["Keyword"] = overall_counts["kw_norm"].map(rep_map)

        st.markdown("#### Overall Top 5 keywords (all documents)")
        st.table(
            overall_counts[["Keyword", "Total hits"]]
            .reset_index(drop=True)
        )

        # Top 3 keywords for each of the top 5 fraud labels
        st.markdown("#### Top 3 keywords for each of the Top 5 fraud labels")

        top_fraud_labels_list = list(top5_labels["fraud_type"])
        rows: List[dict] = []

        for fraud_label in top_fraud_labels_list:
            sub = kw[kw["fraud_type"] == fraud_label].copy()
            if sub.empty:
                continue
            sub_counts = (
                sub.groupby("kw_norm")
                .size()
                .sort_values(ascending=False)
                .head(3)
                .reset_index(name="Hits")
            )
            sub_counts["Keyword"] = sub_counts["kw_norm"].map(rep_map)
            for _, r in sub_counts.iterrows():
                rows.append(
                    {
                        "Fraud label": _pretty_label(fraud_label),
                        "Keyword": r["Keyword"],
                        "Hits": int(r["Hits"]),
                    }
                )

        if rows:
            kw_table = pd.DataFrame(rows)
            st.table(kw_table)
        else:
            st.info("No keyword data aligned to top fraud labels.")

    st.markdown("---")

    # ---------- Trend snapshot for top 3 labels ----------
    st.markdown("### Trend snapshot for top fraud themes")

    top3_labels = list(fraud_hits.head(3).index)
    trend = (
        exploded[exploded["fraud_type"].isin(top3_labels)]
        .groupby(["doc_year", "fraud_type"], dropna=True)["count"]
        .sum()
        .reset_index()
    )

    if trend.empty:
        st.info("No trend data available.")
    else:
        trend_pivot = (
            trend.pivot_table(
                index="doc_year",
                columns="fraud_type",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index()
        )
        trend_pivot.columns = [_pretty_label(c) for c in trend_pivot.columns]
        st.line_chart(trend_pivot, width="stretch")

    st.markdown("---")

    # ---------- Narrative for copy-paste ----------
    st.markdown("### Report-Friendly Narrative")

    with st.expander("Click to copy narrative text for your slide/report", expanded=True):
        st.markdown(
            """
**Data Source Overview**

We built a unified dataset of FinCEN Advisories, Alerts, and Notices published between 1996 and 2025. After scraping and normalizing metadata, we processed all available documents through our FinCEN Fraud Intelligence pipeline. Each PDF is converted into OCRed text with page geometry, tagged with fraud labels using both semantic embeddings and regex-based keyword detection, and summarized into a structured JSON format with high-level summary, primary fraud types, key red flags, and SAR guidance.

**Top Fraud Themes**

Across all publications, money laundering is the dominant theme and appears consistently across the timeline. Terrorist financing is the second most prominent category and shows particular density in the late 2000s and again in the late 2010s. Sanctions evasion and related proliferation financing increase in the late 2010s and early 2020s, reflecting geopolitical shifts. Ransomware and cyber-enabled schemes spike around the COVID era and are often linked to fraud against relief and government programs.

**Solution Approach**

Our solution is an end-to-end FinCEN Fraud Intelligence Platform. It includes web scrapers for all three publication types, a parsing/OCR layer to extract text and geometry, hybrid fraud tagging that combines semantic embeddings with high-precision regex labels, LLM-based summarization into a structured schema, and a Streamlit dashboard that visualizes trends and supports document- and chunk-level exploration. This turns hundreds of static FinCEN PDFs into a living fraud-knowledge hub for analysts.

**Questions for USAA**

1. Which fraud categories from FinCEN’s publications are most operationally painful for your teams today, and how should we prioritize our tagging and dashboard views around those?
2. Would this tool be most valuable as a standalone “FinCEN Fraud Library,” or integrated into existing AML and case management workflows?
3. How useful would it be if a future version of this platform highlighted weak emerging fraud signals before they appear as major FinCEN Alerts, and how might that fit into your risk governance process?
            """
        )


def main():
    st.set_page_config(page_title="FinCEN Fraud Intelligence Explorer", layout="wide")

    st.title("FinCEN Fraud Intelligence Explorer")
    st.write(
        """
        This app sits on top of your OCR + fraud mapping pipeline and lets you:

        - See which fraud types appear most often across FinCEN Advisories / Alerts / Notices
        - Drill into specific documents and read LLM summaries / red flags
        - Explore semantic chunks tagged with fraud labels
        - (New) Auto-generate a Progress Report #3 snapshot for your project
        """
    )

    with st.expander("Where do these data come from?"):
        st.markdown(
            """
            We expect the following CSVs next to this app:

            - `fincen_fraud_mapping.csv` – PDF-level fraud label summary  
            - `fincen_keyword_locations.csv` – exact keyword hits with locations  
            - `fincen_semantic_chunks.csv` – per-chunk semantic fraud labels  
            - `fincen_advisory_summaries.csv` – LLM summaries, red flags, SAR focus  
            """
        )

    dfs = load_data()
    mapping = dfs.get("mapping")
    chunks = dfs.get("chunks")
    keywords = dfs.get("keywords")

    if mapping is None or mapping.empty:
        st.error(
            "No `fincen_fraud_mapping.csv` found next to this app, or it is empty.\n\n"
            "Run `fincen_ocr_fraud_mapper.py` first, then refresh this page."
        )
        return

    exploded = explode_top_labels(mapping)

    if exploded.empty:
        st.warning(
            "No fraud labels found in `top_labels_regex`. "
            "Make sure your mapper is populating regex hits."
        )
        return

    filters = sidebar_filters(exploded)
    exploded_filtered = apply_filters(exploded, filters)

    tab_overview, tab_docs, tab_chunks, tab_progress = st.tabs(
        ["📈 Overview", "📄 Documents", "🧩 Chunks", "📊 Progress Report #3"]
    )

    with tab_overview:
        layout_overview(exploded_filtered, filters["max_fraud_series"])

    with tab_docs:
        layout_documents_table(exploded_filtered, mapping)

    with tab_chunks:
        layout_chunk_explorer(chunks)

    with tab_progress:
        layout_progress_report(exploded, mapping, keywords)


if __name__ == "__main__":
    main()
