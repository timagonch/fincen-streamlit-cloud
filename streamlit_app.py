#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinCEN Fraud Families Timeline + Drill-Down App

This Streamlit app sits on top of the FinCEN pipeline:

  • fincen_fraud_mapping.csv   – semantic backbone from mapper
  • fincen_summaries.json      – LLM summaries per document
  • fincen_summaries.csv       – flat CSV view of summaries (via json_to_csv_summaries.py)

The app focuses on two views:

  1) Fraud Families Over Time
     - How often each fraud family appears by year.
     - Powered by primary_fraud_families + secondary_fraud_families.

  2) Drill-Down Documents
     - Show actual documents matching filters (year range, doc types, fraud families).
     - Surface titles, FinCEN IDs, summaries, and fraud types per document.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import altair as alt
import pandas as pd
import streamlit as st

import json


# ----------------------------- CONFIG ----------------------------------------

MAPPING_CSV = "fincen_fraud_mapping.csv"
SUMMARIES_CSV = "fincen_summaries.csv"

st.set_page_config(
    page_title="FinCEN Fraud Families Timeline",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Data loading ----------------------------------


def load_data():
    """
    Load fincen_summaries.csv (required) and fincen_fraud_mapping.csv (optional).

    Returns:
        summaries_df: one row per document (from summaries CSV)
        docs_df: merged doc-level frame (summaries + mapping if available)
    """
    summaries_path = Path(SUMMARIES_CSV)
    if not summaries_path.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARIES_CSV}. "
            "Run json_to_csv_summaries.py first to generate it."
        )

    summaries_df = pd.read_csv(summaries_path)

    # Parse dates and derive year
    if "doc_date" in summaries_df.columns:
        summaries_df["doc_date_parsed"] = pd.to_datetime(
            summaries_df["doc_date"], errors="coerce"
        )
        summaries_df["year"] = summaries_df["doc_date_parsed"].dt.year
    else:
        summaries_df["doc_date_parsed"] = pd.NaT
        summaries_df["year"] = pd.NA

    # Clean doc_type a bit
    if "doc_type" not in summaries_df.columns:
        summaries_df["doc_type"] = "Unknown"

    # Helper: split pipe-separated lists
    def split_pipe(val) -> List[str]:
        if pd.isna(val):
            return []
        parts = [p.strip() for p in str(val).split("|")]
        return [p for p in parts if p]

    summaries_df["primary_list"] = summaries_df.get(
        "primary_fraud_families", ""
    ).apply(split_pipe)
    summaries_df["secondary_list"] = summaries_df.get(
        "secondary_fraud_families", ""
    ).apply(split_pipe)

    # Union of primary + secondary, per doc
    summaries_df["all_families"] = summaries_df["primary_list"] + summaries_df["secondary_list"]

    # Try to merge mapping CSV if it exists (this adds OCR + location info if present)
    mapping_path = Path(MAPPING_CSV)
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        # Expect article_name to be the join key
        if "article_name" in mapping_df.columns and "article_name" in summaries_df.columns:
            docs_df = mapping_df.merge(
                summaries_df, on="article_name", how="right", suffixes=("", "_map")
            )
        else:
            docs_df = summaries_df.copy()
    else:
        docs_df = summaries_df.copy()

    return summaries_df, docs_df


def explode_families(docs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take docs_df with an 'all_families' list column and return a long frame
    with one row per (doc, fraud_family) combination.
    """
    df = docs_df.copy()
    if "all_families" not in df.columns:
        df["all_families"] = [[] for _ in range(len(df))]

    exploded = df.explode("all_families")
    exploded = exploded[exploded["all_families"].notna() & (exploded["all_families"] != "")]
    exploded = exploded.rename(columns={"all_families": "fraud_family"})
    return exploded


# ----------------------------- UI helpers ------------------------------------


def sidebar_filters(exploded_docs: pd.DataFrame) -> dict:
    """
    Build the sidebar controls and return a dict of selected filters + family mode.
    """
    st.sidebar.header("Filters")

    # Fraud family filter base list
    all_families = sorted(
        f for f in exploded_docs["fraud_family"].dropna().unique() if f
    )

    family_mode = st.sidebar.radio(
        "Fraud family mode",
        options=["Manual selection", "Top N overall", "Top N per year"],
        index=0,
        help=(
            "- **Manual selection**: Pick one or more fraud families.\n"
            "- **Top N overall**: Take the top N families (by doc count over the entire range).\n"
            "- **Top N per year**: For each year, pick the top N families independently."
        ),
    )

    selected_families: List[str] = []
    top_n: int = 10

    if family_mode == "Manual selection":
        selected_families = st.sidebar.multiselect(
            "Fraud families",
            options=all_families,
            default=all_families[:10],
        )
    else:
        # For Top N variants, let user pick N based on range of list length
        max_n = len(all_families)
        if max_n == 0:
            max_n = 1
        default_n = min(10, max_n) if max_n else 3
        top_n = st.sidebar.slider(
            "Number of fraud families to show (Top N)",
            min_value=1,
            max_value=max_n or 1,
            value=default_n or 1,
            step=1,
        )

    # Publication type filter
    all_doc_types = sorted(
        d for d in exploded_docs.get("doc_type", pd.Series(dtype=str)).dropna().unique() if d
    )
    if not all_doc_types:
        all_doc_types = ["Advisory", "Alert", "Notice"]
    selected_doc_types = st.sidebar.multiselect(
        "Publication types",
        options=all_doc_types,
        default=all_doc_types,
    )

    # Year range filter
    if "year" in exploded_docs.columns and exploded_docs["year"].notna().any():
        min_year = int(exploded_docs["year"].min())
        max_year = int(exploded_docs["year"].max())
        year_range = st.sidebar.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )
    else:
        year_range = None

    filters = {
        "family_mode": family_mode,
        "selected_families": selected_families,
        "top_n": top_n,
        "selected_doc_types": selected_doc_types,
        "year_range": year_range,
    }
    return filters


def apply_filters_base(exploded_docs: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply only year + doc_type filters, not fraud family filters.
    """
    df = exploded_docs.copy()

    # Filter by doc_type
    doc_types = filters.get("selected_doc_types")
    if doc_types:
        df = df[df["doc_type"].isin(doc_types)]

    # Filter by year range
    year_range = filters.get("year_range")
    if year_range and "year" in df.columns:
        y_min, y_max = year_range
        df = df[(df["year"] >= y_min) & (df["year"] <= y_max)]

    return df


def filter_exploded_with_family_mode(
    exploded_docs: pd.DataFrame, filters: dict
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply filters to exploded_docs depending on the selected fraud family mode.

    Returns:
        filtered_exploded: exploded_docs filtered by year, doc_type, and family logic
        active_families: list of fraud families that are still in play
    """
    mode = filters.get("family_mode", "Manual selection")

    # First apply only the base filters
    base = apply_filters_base(exploded_docs, filters)
    if base.empty:
        return base, []

    if mode == "Manual selection":
        selected = filters.get("selected_families") or []
        if not selected:
            # If user didn't select any families, just return empty
            return base.iloc[0:0].copy(), []
        filtered = base[base["fraud_family"].isin(selected)]
        active_families = sorted(
            f for f in filtered["fraud_family"].dropna().unique() if f
        )
        return filtered, active_families

    # For Top N modes, we determine families by doc counts
    counts = (
        base.groupby("fraud_family", dropna=True)["article_name"]
        .nunique()
        .reset_index(name="docs")
    )
    counts = counts[counts["fraud_family"].notna() & (counts["fraud_family"] != "")]
    if counts.empty:
        return base.iloc[0:0].copy(), []

    top_n = max(1, int(filters.get("top_n", 10)))
    counts = counts.sort_values("docs", ascending=False)
    top_families_all = counts["fraud_family"].head(top_n).tolist()

    if mode == "Top N overall":
        filtered = base[base["fraud_family"].isin(top_families_all)]
        active_families = sorted(
            f for f in filtered["fraud_family"].dropna().unique() if f
        )
        return filtered, active_families

    # For "Top N per year", the per-year Top N logic happens in the chart tab.
    # Here we only apply base filters and report all families in that slice.
    return base, sorted(
        f for f in base["fraud_family"].dropna().unique() if f
    )


# ----------------------------- Selection helpers -----------------------------


def extract_year_and_family_from_event(
    event,
    selection_name: str = "fraud_point",
) -> Tuple[Optional[int], Optional[str]]:
    """
    Robustly parse the Streamlit Altair selection event and return (year, fraud_family).

    Handles the main formats Streamlit can emit for a point selection:
      - event.selection[selection_name] is a dict with "year"/"fraud_family"
      - or a dict with "fields" + "values"
      - or a dict with "values": [ { "year": ..., "fraud_family": ... }, ... ]
    """
    if not event:
        return None, None

    # event can be a custom dict-like object; support both attribute and item access.
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")

    if not sel:
        return None, None

    selection_obj = getattr(sel, selection_name, None)
    if selection_obj is None and isinstance(sel, dict):
        selection_obj = sel.get(selection_name)

    if not selection_obj:
        return None, None

    year = None
    family = None

    # Case 1: direct dict with year / fraud_family
    if isinstance(selection_obj, dict):
        if "year" in selection_obj or "fraud_family" in selection_obj:
            year = selection_obj.get("year")
            family = selection_obj.get("fraud_family")

        # Case 2: dict with fields + values
        if (year is None or family is None) and "fields" in selection_obj and "values" in selection_obj:
            fields = selection_obj.get("fields")
            values = selection_obj.get("values")
            if isinstance(fields, list) and isinstance(values, list) and len(fields) == len(values):
                mapping = dict(zip(fields, values))
                year = year or mapping.get("year")
                family = family or mapping.get("fraud_family")

        # Case 3: dict with "values" -> list of point dicts
        if (year is None or family is None) and isinstance(selection_obj.get("values"), list):
            first = selection_obj["values"][0] if selection_obj["values"] else None
            if isinstance(first, dict):
                year = year or first.get("year")
                family = family or first.get("fraud_family")

    # Case 4: list of dicts
    elif isinstance(selection_obj, list) and selection_obj:
        first = selection_obj[0]
        if isinstance(first, dict):
            year = first.get("year")
            family = first.get("fraud_family")

    # Make sure year is an int if present
    try:
        if year is not None:
            year = int(year)
    except Exception:
        pass

    return year, family


# ----------------------------- Tabs ------------------------------------------


def tab_fraud_families_over_time(exploded_docs: pd.DataFrame, filters: dict):
    st.subheader("Fraud Families Over Time")

    st.markdown(
        """
        This view shows how often each **fraud family** appears in FinCEN publications
        over time. A document contributes to a family in a given year if that family
        is present in either its **primary** or **secondary** fraud families.

        Counts are aggregated across Advisories, Alerts, and Notices.

        👉 Click a bar in the chart to see the **documents** for that (year, fraud family).
        """
    )

    mode = filters.get("family_mode", "Manual selection")
    top_n = max(1, int(filters.get("top_n", 10)))

    # Helper to render a drill-down table of documents for a selected bar.
    def _render_drilldown_table(
        source_df: pd.DataFrame,
        selected_year: Optional[int],
        selected_family: Optional[str],
        label_suffix: str = "",
    ):
        """Show docs + LLM summary for a selected (year, fraud_family) bar."""
        if selected_year is None or not selected_family:
            return

        # Filter down to this (year, fraud_family)
        focus = source_df[
            (source_df["year"] == selected_year)
            & (source_df["fraud_family"] == selected_family)
        ].copy()

        if focus.empty:
            st.info("No documents found for this selection.")
            return

        # One row per document
        focus_docs = focus.drop_duplicates(subset=["article_name"])

        # Little badge with count
        num_docs = focus_docs["article_name"].nunique()
        badge_text = (
            f"Showing {num_docs} document{'s' if num_docs != 1 else ''} "
            f"for {selected_family} ({selected_year})"
        )
        st.markdown(
            f"<div style='display:inline-block;padding:0.25rem 0.6rem;"
            f"border-radius:999px;background-color:#1d4ed8;color:white;"
            f"font-size:0.85rem;margin-bottom:0.4rem;'>{badge_text}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"### Articles for **{selected_family}** in **{selected_year}**{label_suffix}"
        )

        # Table view (subset of columns)
        display_cols = [
            "doc_date",
            "year",
            "fincen_id",
            "doc_type",
            "doc_title",
            "article_name",
            "primary_fraud_families",
            "secondary_fraud_families",
        ]
        available_cols = [c for c in display_cols if c in focus_docs.columns]

        table_df = (
            focus_docs[available_cols]
            .sort_values(["doc_date", "article_name"], na_position="last")
            .reset_index(drop=True)
        )
        st.dataframe(table_df, use_container_width=True)

        # --- Article-level LLM summary viewer ---
        # Use full focus (all columns) so we keep high_level_summary, key_red_flags, etc.
        focus_docs_sorted = (
            focus.sort_values(["doc_date", "article_name"], na_position="last")
            .reset_index(drop=True)
        )

        if focus_docs_sorted.empty:
            return

        st.markdown("#### View LLM summary for a specific article")

        option_labels = []
        for i, row in focus_docs_sorted.iterrows():
            label = (
                f"{row.get('doc_date', '')} – "
                f"{row.get('fincen_id', '')} – "
                f"{row.get('doc_title', '') or row.get('article_name', '')}"
            )
            option_labels.append(label)

        selected_idx = st.selectbox(
            "Choose an article:",
            options=list(range(len(option_labels))),
            format_func=lambda i: option_labels[i],
            key=f"summary_select_{selected_year}_{selected_family}",
        )

        selected_row = focus_docs_sorted.iloc[selected_idx]

        st.markdown("##### LLM-generated summary")

        # High-level summary
        if "high_level_summary" in focus_docs_sorted.columns and pd.notna(
            selected_row.get("high_level_summary", None)
        ):
            st.markdown("**High-level summary**")
            st.write(selected_row["high_level_summary"])


        # Primary fraud families (bullet list)
        if pd.notna(selected_row.get("primary_fraud_families", None)):
            st.markdown("**Primary fraud families**")
            raw_primary = str(selected_row["primary_fraud_families"])
            primary_list = [p.strip() for p in raw_primary.split("|") if p.strip()]
            if primary_list:
                st.markdown("\n".join(f"- {p}" for p in primary_list))
            else:
                st.write(raw_primary)

        # Secondary fraud families (bullet list)
        if pd.notna(selected_row.get("secondary_fraud_families", None)):
            st.markdown("**Secondary fraud families**")
            raw_secondary = str(selected_row["secondary_fraud_families"])
            secondary_list = [s.strip() for s in raw_secondary.split("|") if s.strip()]
            if secondary_list:
                st.markdown("\n".join(f"- {s}" for s in secondary_list))
            else:
                st.write(raw_secondary)


        # Key red flags
        if "key_red_flags" in focus_docs_sorted.columns and pd.notna(
            selected_row.get("key_red_flags", None)
        ):
            st.markdown("**Key red flags**")
            raw_flags = str(selected_row["key_red_flags"])
            flags = [f.strip() for f in raw_flags.split(" | ") if f.strip()]
            if flags:
                st.markdown("\n".join(f"- {flag}" for flag in flags))
            else:
                st.write(raw_flags)
        # Specific schemes (from LLM) – show in a collapsible section
        # Prefer the structured JSON if available; fall back to the flat string.
        has_json = (
            "specific_schemes_json" in focus_docs_sorted.columns
            and pd.notna(selected_row.get("specific_schemes_json", None))
        )
        has_flat = (
            "specific_schemes_flat" in focus_docs_sorted.columns
            and pd.notna(selected_row.get("specific_schemes_flat", None))
        )

        if has_json or has_flat:
            with st.expander("Specific schemes (family → scheme → notes)", expanded=False):
                if has_json:
                    try:
                        schemes = json.loads(selected_row["specific_schemes_json"])
                    except Exception:
                        schemes = None

                    if isinstance(schemes, list) and schemes:
                        schemes_df = pd.DataFrame(schemes)

                        # Nice column names if keys exist
                        col_renames = {
                            "family": "Fraud family",
                            "label": "Scheme label",
                            "notes": "Notes",
                        }
                        schemes_df = schemes_df.rename(
                            columns={k: v for k, v in col_renames.items() if k in schemes_df.columns}
                        )

                        st.dataframe(schemes_df, use_container_width=True)
                    else:
                        st.write("No structured schemes available.")
                elif has_flat:
                    # Fallback: simple bullet list from the flat pipe-separated string
                    items = [
                        s.strip()
                        for s in str(selected_row["specific_schemes_flat"]).split("|")
                        if s.strip()
                    ]
                    if items:
                        st.markdown("\n".join(f"- {item}" for item in items))
                    else:
                        st.write("No specific schemes listed.")
            



    # --- SPECIAL HANDLING FOR "Top N per year" ---
    if mode == "Top N per year":
        # Apply only year + doc_type filters first
        base = apply_filters_base(exploded_docs, filters)
        if base.empty:
            st.info("No documents match the current filters.")
            return

        # Count distinct docs per (year, fraud_family)
        counts = (
            base.groupby(["year", "fraud_family"], dropna=True)["article_name"]
            .nunique()
            .reset_index(name="docs")
        )
        if counts.empty:
            st.info("No document counts available for the current filters.")
            return

        # Rank within each year and keep only Top N *for that year*
        counts["rank"] = counts.groupby("year")["docs"].rank(
            method="first", ascending=False
        )
        counts_top = counts[counts["rank"] <= top_n]

        if counts_top.empty:
            st.info("No fraud families fall into the Top N per year for this range.")
            return

        st.caption(
            f"For each year, showing fraud families that are in the Top {top_n} "
            f"by document count for that year. Families can change year-to-year."
        )

        selector = alt.selection_point(
            "fraud_point",
            fields=["year", "fraud_family"],
        )

        chart = (
            alt.Chart(counts_top)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("docs:Q", title="Number of documents"),
                color=alt.Color("fraud_family:N", title="Fraud family"),
                opacity=alt.condition(selector, alt.value(1.0), alt.value(0.3)),
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),
                    alt.Tooltip("fraud_family:N", title="Fraud family"),
                    alt.Tooltip("docs:Q", title="# of documents"),
                ],
            )
            .add_params(selector)
            .properties(height=450)
        )

        event = st.altair_chart(
            chart,
            width="stretch",
            on_select="rerun",
            selection_mode="fraud_point",
            key="fraud_families_chart_top_per_year",
        )

        selected_year, selected_family = extract_year_and_family_from_event(
            event, selection_name="fraud_point"
        )

        # Drill-down table for the selected bar
        _render_drilldown_table(
            base, selected_year, selected_family, label_suffix=" (Top N per year mode)"
        )
        return

    # --- Manual selection / Top N overall ---
    filtered, active_families = filter_exploded_with_family_mode(exploded_docs, filters)

    if filtered.empty or not active_families:
        st.info("No documents match the current filters.")
        return

    st.caption(
        "Only documents whose primary or secondary fraud families intersect with the "
        "selected families (or Top N overall) are included below."
    )

    # Count unique documents per (year, fraud_family)
    counts = (
        filtered.groupby(["year", "fraud_family"], dropna=True)
        .agg(docs=("article_name", "nunique"))
        .reset_index()
    )

    if counts.empty:
        st.info("No document counts available for the current filters.")
        return

    selector = alt.selection_point(
        "fraud_point",
        fields=["year", "fraud_family"],
    )

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("docs:Q", title="Number of documents"),
            color=alt.Color("fraud_family:N", title="Fraud family"),
            opacity=alt.condition(selector, alt.value(1.0), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("fraud_family:N", title="Fraud family"),
                alt.Tooltip("docs:Q", title="# of documents"),
            ],
        )
        .add_params(selector)
        .properties(height=450)
    )

    event = st.altair_chart(
        chart,
        width="stretch",
        on_select="rerun",
        selection_mode="fraud_point",
        key="fraud_families_chart_manual_top_overall",
    )

    selected_year, selected_family = extract_year_and_family_from_event(
        event, selection_name="fraud_point"
    )

    # Drill-down table driven by the chart selection
    _render_drilldown_table(
        filtered,
        selected_year,
        selected_family,
        label_suffix=" (Manual / Top N overall mode)",
    )



# ----------------------------- Main app --------------------------------------
def main():
    st.title("FinCEN Fraud Families Timeline & Drill-Down")

    st.markdown(
        """
        This app visualizes FinCEN Advisories, Alerts, and Notices using
        **fraud families** detected by the semantic mapper + LLM summaries.

        - Use the **left sidebar** to filter by year range, publication types,
          and fraud family mode.
        - Click any bar in the chart to see the documents and LLM insights
          for that (year, fraud family).
        """
    )

    try:
        summaries_df, docs_df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    exploded = explode_families(docs_df)

    filters = sidebar_filters(exploded)

    # Single main view: chart + drill-down + LLM panel
    tab_fraud_families_over_time(exploded, filters)

if __name__ == "__main__":
    main()