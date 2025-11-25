#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinCEN Fraud Families Timeline + Drill-Down App (Supabase version)

This Streamlit app sits on top of the new FinCEN pipeline:

  • fincen_publications      – metadata about each publication
  • fincen_llm_summaries     – LLM summaries + fraud families/schemes per doc

The app focuses on a single main view:

  1) Fraud Families Over Time
     - How often each fraud family appears by year.
     - Powered by primary_fraud_families + secondary_fraud_families.

  2) Drill-Down Documents
     - Show actual documents matching filters (year range, doc types, fraud families).
     - Surface titles, FinCEN IDs, summaries, fraud types, red flags, and schemes.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict

import altair as alt
import pandas as pd
import streamlit as st
import json

from supabase_helpers import get_supabase_client


st.set_page_config(
    page_title="FinCEN Fraud Families Timeline",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading from Supabase
# ---------------------------------------------------------------------------


def _ensure_list(val: Any) -> List[Any]:
    """
    Make sure JSONB-ish fields become Python lists.

    Handles:
      - already-a-list
      - None / NaN
      - JSON-encoded strings
      - pipe-separated strings (last-resort)
    """
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    # Try JSON decode
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            # Maybe it was pipe-separated text
            parts = [p.strip() for p in s.split("|")]
            return [p for p in parts if p]
    # Fallback
    return [val]


def load_data_from_supabase() -> pd.DataFrame:
    """
    Load summaries + publication metadata from Supabase and return a single
    DataFrame with one row per document.

    Columns will include:
      - doc_key
      - doc_title
      - doc_date
      - year
      - doc_type
      - fincen_id (if present)
      - high_level_summary
      - primary_fraud_families (list)
      - secondary_fraud_families (list)
      - specific_schemes (list[dict])
      - key_red_flags (list[str])
    """
    client = get_supabase_client()

    # Summaries table (LLM output)
    summaries_resp = client.table("fincen_llm_summaries").select(
        "doc_key, title, doc_type, date, "
        "high_level_summary, "
        "primary_fraud_families, secondary_fraud_families, "
        "specific_schemes, key_red_flags"
    ).execute()
    summaries_rows = summaries_resp.data or []

    if not summaries_rows:
        return pd.DataFrame()

    summaries_df = pd.DataFrame(summaries_rows)

    # Normalize column names
    summaries_df = summaries_df.rename(
        columns={
            "title": "doc_title",
            "date": "doc_date",
        }
    )

    # Parse dates and derive year
    summaries_df["doc_date_parsed"] = pd.to_datetime(
        summaries_df["doc_date"], errors="coerce"
    )
    summaries_df["year"] = summaries_df["doc_date_parsed"].dt.year

    # Publications table (metadata)
    pubs_resp = client.table("fincen_publications").select(
        "doc_key, fincen_id, pdf_filename, pdf_url, detail_url, doc_type"
    ).execute()
    pubs_rows = pubs_resp.data or []
    pubs_df = pd.DataFrame(pubs_rows) if pubs_rows else pd.DataFrame(
        columns=["doc_key", "fincen_id", "pdf_filename", "pdf_url", "detail_url", "doc_type"]
    )

    # Merge on doc_key (summaries are the primary set)
    docs_df = summaries_df.merge(
        pubs_df,
        on="doc_key",
        how="left",
        suffixes=("", "_pub"),
    )

    # If doc_type is missing in summaries, use publications doc_type
    if "doc_type" in docs_df.columns and "doc_type_pub" in docs_df.columns:
        docs_df["doc_type"] = docs_df["doc_type"].fillna(docs_df["doc_type_pub"])
        docs_df = docs_df.drop(columns=["doc_type_pub"])

    # Ensure fraud family lists
    docs_df["primary_fraud_families"] = docs_df["primary_fraud_families"].apply(
        _ensure_list
    )
    docs_df["secondary_fraud_families"] = docs_df["secondary_fraud_families"].apply(
        _ensure_list
    )

    # Union of primary + secondary per doc
    docs_df["all_families"] = docs_df["primary_fraud_families"] + docs_df[
        "secondary_fraud_families"
    ]

    # Ensure key_red_flags & specific_schemes exist
    if "key_red_flags" not in docs_df.columns:
        docs_df["key_red_flags"] = [[] for _ in range(len(docs_df))]
    else:
        docs_df["key_red_flags"] = docs_df["key_red_flags"].apply(_ensure_list)

    if "specific_schemes" not in docs_df.columns:
        docs_df["specific_schemes"] = [[] for _ in range(len(docs_df))]

    return docs_df


def explode_families(docs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take docs_df with an 'all_families' list column and return a long frame
    with one row per (doc, fraud_family) combination.
    """
    df = docs_df.copy()
    if "all_families" not in df.columns:
        df["all_families"] = [[] for _ in range(len(df))]

    exploded = df.explode("all_families")
    exploded = exploded[
        exploded["all_families"].notna() & (exploded["all_families"] != "")
    ]
    exploded = exploded.rename(columns={"all_families": "fraud_family"})
    return exploded


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def sidebar_filters(docs_df: pd.DataFrame) -> dict:
    """
    Build the sidebar controls and return a dict of selected filters + modes.
    We use docs_df (one row per doc); the actual exploded view is built later
    depending on the "primary only" vs "primary + secondary" toggle.
    """
    st.sidebar.header("Filters")

    # Toggle: count only primary families vs primary+secondary
    family_source_mode = st.sidebar.radio(
        "Which fraud families to count?",
        options=["Primary only", "Primary + Secondary"],
        index=1,
        help=(
            "• Primary only: each document contributes only to its primary_fraud_families.\n"
            "• Primary + Secondary: a document contributes to any family listed as primary "
            "or secondary (this was the original behavior)."
        ),
    )

    # Determine which list to use for building the family picker
    if family_source_mode == "Primary only":
        family_col = "primary_fraud_families"
    else:
        family_col = "all_families"

    fam_series = docs_df[family_col].explode().dropna()
    all_families = sorted({f for f in fam_series if f})

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
        max_n = len(all_families) or 1
        default_n = min(10, max_n)
        top_n = st.sidebar.slider(
            "Number of fraud families to show (Top N)",
            min_value=1,
            max_value=max_n,
            value=default_n,
            step=1,
        )

    # Publication type filter
    all_doc_types = sorted(
        d for d in docs_df.get("doc_type", pd.Series(dtype=str)).dropna().unique() if d
    )
    if not all_doc_types:
        all_doc_types = ["advisory", "alert", "notice"]
    selected_doc_types = st.sidebar.multiselect(
        "Publication types",
        options=all_doc_types,
        default=all_doc_types,
    )

    # Year range filter
    if "year" in docs_df.columns and docs_df["year"].notna().any():
        min_year = int(docs_df["year"].min())
        max_year = int(docs_df["year"].max())
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
        "family_source_mode": family_source_mode,  # primary-only vs primary+secondary
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
        return base.iloc[0:0].copy(), []

    if mode == "Manual selection":
        selected = filters.get("selected_families") or []
        if not selected:
            return base.iloc[0:0].copy(), []
        filtered = base[base["fraud_family"].isin(selected)]
        active_families = sorted(
            f for f in filtered["fraud_family"].dropna().unique() if f
        )
        return filtered, active_families

    # For Top N modes, we determine families by doc counts
    counts = (
        base.groupby("fraud_family", dropna=True)["doc_key"]
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


# ---------------------------------------------------------------------------
# Selection helpers (Altair → Streamlit)
# ---------------------------------------------------------------------------


def extract_year_and_family_from_event(
    event,
    selection_name: str = "fraud_point",
) -> Tuple[Optional[int], Optional[str]]:
    """
    Robustly parse the Streamlit Altair selection event and return (year, fraud_family).
    """
    if not event:
        return None, None

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

    if isinstance(selection_obj, dict):
        if "year" in selection_obj or "fraud_family" in selection_obj:
            year = selection_obj.get("year")
            family = selection_obj.get("fraud_family")

        if (year is None or family is None) and "fields" in selection_obj and "values" in selection_obj:
            fields = selection_obj.get("fields")
            values = selection_obj.get("values")
            if isinstance(fields, list) and isinstance(values, list) and len(fields) == len(values):
                mapping = dict(zip(fields, values))
                year = year or mapping.get("year")
                family = family or mapping.get("fraud_family")

        if (year is None or family is None) and isinstance(selection_obj.get("values"), list):
            first = selection_obj["values"][0] if selection_obj["values"] else None
            if isinstance(first, dict):
                year = year or first.get("year")
                family = family or first.get("fraud_family")

    elif isinstance(selection_obj, list) and selection_obj:
        first = selection_obj[0]
        if isinstance(first, dict):
            year = first.get("year")
            family = first.get("fraud_family")

    try:
        if year is not None:
            year = int(year)
    except Exception:
        pass

    return year, family


# ---------------------------------------------------------------------------
# Main chart + drill-down tab
# ---------------------------------------------------------------------------


def tab_fraud_families_over_time(exploded_docs: pd.DataFrame, filters: dict):
    st.subheader("Fraud Families Over Time")

    src_mode = filters.get("family_source_mode", "Primary + Secondary")
    if src_mode == "Primary only":
        src_caption = (
            "Each document contributes only to the fraud families listed as **primary**."
        )
    else:
        src_caption = (
            "Each document contributes to any fraud family listed as **primary or secondary**."
        )

    st.markdown(
        f"""
        This view shows how often each **fraud family** appears in FinCEN publications
        over time.

        {src_caption}

        Counts are aggregated across Advisories, Alerts, and Notices.

        👉 Click a bar in the chart to see the **documents** for that (year, fraud family).
        """
    )

    mode = filters.get("family_mode", "Manual selection")
    top_n = max(1, int(filters.get("top_n", 10)))

    def _render_drilldown_table(
        source_df: pd.DataFrame,
        selected_year: Optional[int],
        selected_family: Optional[str],
        label_suffix: str = "",
    ):
        """Show docs + LLM summary for a selected (year, fraud_family) bar."""
        if selected_year is None or not selected_family:
            return

        focus = source_df[
            (source_df["year"] == selected_year)
            & (source_df["fraud_family"] == selected_family)
        ].copy()

        if focus.empty:
            st.info("No documents found for this selection.")
            return

        focus_docs = focus.drop_duplicates(subset=["doc_key"])

        num_docs = focus_docs["doc_key"].nunique()
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

        display_cols = [
            "doc_date",
            "year",
            "fincen_id",
            "doc_type",
            "doc_title",
            "doc_key",
            "primary_fraud_families",
            "secondary_fraud_families",
        ]
        available_cols = [c for c in display_cols if c in focus_docs.columns]

        table_df = (
            focus_docs[available_cols]
            .sort_values(["doc_date", "doc_key"], na_position="last")
            .reset_index(drop=True)
        )
        st.dataframe(table_df, use_container_width=True)

        # Article-level summary viewer
        focus_docs_sorted = (
            focus.sort_values(["doc_date", "doc_key"], na_position="last")
            .reset_index(drop=True)
        )
        if focus_docs_sorted.empty:
            return

        st.markdown("#### View LLM summary for a specific article")

        option_labels = []
        for _, row in focus_docs_sorted.iterrows():
            label = (
                f"{row.get('doc_date', '')} – "
                f"{row.get('doc_type', '')} – "
                f"{row.get('doc_title', '') or row.get('doc_key', '')}"
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

        if pd.notna(selected_row.get("high_level_summary", None)):
            st.markdown("**High-level summary**")
            st.write(selected_row["high_level_summary"])

        # Primary fraud families (bullet list)
        primary_list = _ensure_list(selected_row.get("primary_fraud_families"))
        if primary_list:
            st.markdown("**Primary fraud families**")
            st.markdown("\n".join(f"- {p}" for p in primary_list))

        # Secondary fraud families (bullet list)
        secondary_list = _ensure_list(selected_row.get("secondary_fraud_families"))
        if secondary_list:
            st.markdown("**Secondary fraud families**")
            st.markdown("\n".join(f"- {s}" for s in secondary_list))

        # Key red flags
        flags = _ensure_list(selected_row.get("key_red_flags"))
        if flags:
            st.markdown("**Key red flags**")
            st.markdown("\n".join(f"- {flag}" for flag in flags))

        # Specific schemes (JSON list of dicts)
        schemes = selected_row.get("specific_schemes")
        if isinstance(schemes, str):
            try:
                schemes = json.loads(schemes)
            except Exception:
                schemes = None

        if isinstance(schemes, list) and schemes:
            with st.expander("Specific schemes (family → scheme → notes)", expanded=False):
                try:
                    schemes_df = pd.DataFrame(schemes)
                except Exception:
                    schemes_df = None

                if schemes_df is not None and not schemes_df.empty:
                    col_renames = {
                        "fraud_family": "Fraud family",
                        "scheme_label": "Scheme label",
                        "description": "Description",
                        "notes": "Notes",
                    }
                    schemes_df = schemes_df.rename(
                        columns={
                            k: v
                            for k, v in col_renames.items()
                            if k in schemes_df.columns
                        }
                    )
                    st.dataframe(schemes_df, use_container_width=True)
                else:
                    st.write("No structured schemes available.")

    # Top N per year mode
    if mode == "Top N per year":
        base = apply_filters_base(exploded_docs, filters)
        if base.empty:
            st.info("No documents match the current filters.")
            return

        counts = (
            base.groupby(["year", "fraud_family"], dropna=True)["doc_key"]
            .nunique()
            .reset_index(name="docs")
        )
        if counts.empty:
            st.info("No document counts available for the current filters.")
            return

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

        _render_drilldown_table(
            base, selected_year, selected_family, label_suffix=" (Top N per year mode)"
        )
        return

    # Manual / Top N overall
    filtered, active_families = filter_exploded_with_family_mode(exploded_docs, filters)

    if filtered.empty or not active_families:
        st.info("No documents match the current filters.")
        return

    st.caption(
        "Only documents whose primary or secondary fraud families intersect with the "
        "selected families (or Top N overall) are included below."
    )

    counts = (
        filtered.groupby(["year", "fraud_family"], dropna=True)
        .agg(docs=("doc_key", "nunique"))
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

    _render_drilldown_table(
        filtered,
        selected_year,
        selected_family,
        label_suffix=" (Manual / Top N overall mode)",
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.title("FinCEN Fraud Families Timeline & Drill-Down")

    st.markdown(
        """
        This app visualizes FinCEN Advisories, Alerts, and Notices using
        **fraud families** extracted from the LLM summaries stored in Supabase.

        - Use the **left sidebar** to filter by year range, publication types,
          fraud family mode, and whether to count only primary fraud families
          or both primary and secondary.
        - Click any bar in the chart to see the documents and LLM insights
          for that (year, fraud family).
        """
    )

    docs_df = load_data_from_supabase()
    if docs_df.empty:
        st.warning(
            "No data found. Make sure fincen_llm_summaries and fincen_publications "
            "are populated in Supabase."
        )
        return

    # Build filters FIRST (including the primary vs primary+secondary toggle)
    filters = sidebar_filters(docs_df)

    # Depending on the toggle, set which families we explode for counting
    if filters.get("family_source_mode") == "Primary only":
        docs_for_explode = docs_df.copy()
        docs_for_explode["all_families"] = docs_for_explode["primary_fraud_families"]
    else:
        docs_for_explode = docs_df

    exploded = explode_families(docs_for_explode)
    if exploded.empty:
        st.warning(
            "No fraud family labels found after applying the selected mode. "
            "Check that primary_fraud_families and secondary_fraud_families "
            "are present in fincen_llm_summaries."
        )
        return

    tab_fraud_families_over_time(exploded, filters)


if __name__ == "__main__":
    main()
