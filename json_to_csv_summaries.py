import json
import pandas as pd
from pathlib import Path

# ---- CONFIG ----
# This is your LLM output file that includes advisories, alerts, and notices
INPUT_JSON = Path("fincen_summaries.json")     # <-- use your exact filename here
OUTPUT_CSV = Path("fincen_advisory_summaries.csv")  # name is fine even though it has all 3 types


def join_list(value):
    """
    Convert a list of strings to a single pipe-separated string for CSV.
    If it's already a string or None, just return it as-is.
    """
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return value


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Could not find {INPUT_JSON.resolve()}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    # Expected structure:
    # {
    #   "advis11.pdf": { ... },
    #   "FIN-2024-NTC1.pdf": { ... },
    #   "alert123.pdf": { ... },
    #   ...
    # }
    if not isinstance(data, dict):
        raise ValueError(
            "Expected top-level JSON to be a dict mapping 'pdf_name' -> payload dict."
        )

    for pdf_filename, payload in data.items():
        if not isinstance(payload, dict):
            # Skip weird entries
            continue

        row = {
            # Key we will join on in Streamlit with fincen_advisories.csv / mapping
            "pdf_filename": pdf_filename,
            # Duplicate for clarity
            "article_name": payload.get("article_name", pdf_filename),
            "fincen_id": payload.get("fincen_id", ""),
            "doc_type": payload.get("doc_type", ""),      # Advisory / Alert / Notice
            "doc_title": payload.get("doc_title", ""),
            "doc_date": payload.get("doc_date", ""),
            "high_level_summary": payload.get("high_level_summary", ""),
            # Lists -> pipe-separated strings
            "primary_fraud_types": join_list(payload.get("primary_fraud_types", [])),
            "key_red_flags": join_list(payload.get("key_red_flags", [])),
            "recommended_sar_focus": join_list(
                payload.get("recommended_sar_focus", [])
            ),
            "why_it_matters_for_usaa": payload.get("why_it_matters_for_usaa", ""),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Optional: enforce column order
    cols = [
        "pdf_filename",
        "article_name",
        "fincen_id",
        "doc_type",
        "doc_title",
        "doc_date",
        "high_level_summary",
        "primary_fraud_types",
        "key_red_flags",
        "recommended_sar_focus",
        "why_it_matters_for_usaa",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
