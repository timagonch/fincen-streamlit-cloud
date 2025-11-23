import json
import pandas as pd
from pathlib import Path

# ---- CONFIG ----
# This is your LLM output file that includes advisories, alerts, and notices
INPUT_JSON = Path("fincen_summaries.json")     # can be overridden if needed
OUTPUT_CSV = Path("fincen_summaries.csv")      # flat, dashboard-friendly CSV


def join_list(value):
    """
    Convert a list of strings to a single pipe-separated string for CSV.
    If it's already a string or None, just return it as-is.
    """
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return value


def flatten_specific_schemes(schemes):
    """
    Flatten the list of scheme dicts into two representations:
      - human_readable: 'family: label' pipe-separated
      - json_str: full JSON string for drill-down use

    schemes is expected to look like:
      [
        {"family": "human_trafficking", "label": "...", "notes": "..."},
        ...
      ]
    """
    if not schemes:
        return "", "[]"

    if not isinstance(schemes, list):
        return str(schemes), json.dumps(schemes, ensure_ascii=False)

    human_bits = []
    for s in schemes:
        if not isinstance(s, dict):
            human_bits.append(str(s))
            continue
        family = s.get("family") or ""
        label = s.get("label") or ""
        if family and label:
            human_bits.append(f"{family}: {label}")
        elif label:
            human_bits.append(label)
        elif family:
            human_bits.append(family)

    human_readable = " | ".join(human_bits)
    json_str = json.dumps(schemes, ensure_ascii=False)
    return human_readable, json_str


def load_summaries(path: Path):
    """
    Load summaries JSON. We support two shapes:

    1) Dict keyed by article_name:
         {
           "Advisory%20EIP%20FINAL%20508.pdf": {...},
           ...
         }

    2) List of summary dicts:
         [
           {"article_name": "...", ...},
           ...
         ]
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    if isinstance(data, dict):
        # Newer shape: keyed by article_name
        for article_name, rec in data.items():
            if isinstance(rec, dict):
                rec = dict(rec)  # shallow copy
                # Ensure article_name present as a column
                rec.setdefault("article_name", article_name)
                rows.append(rec)
            else:
                rows.append({"article_name": article_name, "raw": rec})
    elif isinstance(data, list):
        # Older shape: list of rows (already flat-ish)
        for rec in data:
            if isinstance(rec, dict):
                rows.append(rec)
            else:
                rows.append({"raw": rec})
    else:
        # Unexpected shape; wrap it
        rows.append({"raw": data})

    return rows


def main():
    rows = load_summaries(INPUT_JSON)

    # Normalize records and add flattened columns
    normalized_rows = []
    for rec in rows:
        rec = dict(rec)  # shallow copy so we can modify

        # Normalize list fields to pipe-separated strings
        rec["primary_fraud_families"] = join_list(
            rec.get("primary_fraud_families") or rec.get("primary_fraud_types")
        )
        rec["secondary_fraud_families"] = join_list(
            rec.get("secondary_fraud_families")
        )
        rec["key_red_flags"] = join_list(rec.get("key_red_flags"))

        # Flatten specific_schemes into two columns
        human_schemes, json_schemes = flatten_specific_schemes(
            rec.get("specific_schemes")
        )
        rec["specific_schemes_flat"] = human_schemes
        rec["specific_schemes_json"] = json_schemes

        normalized_rows.append(rec)

    df = pd.DataFrame(normalized_rows)

    # --- Column ordering for the CSV ---
    cols = [
        "article_name",
        "fincen_id",
        "doc_type",
        "doc_title",
        "doc_date",
        "high_level_summary",
        "primary_fraud_families",
        "secondary_fraud_families",
        "specific_schemes_flat",
        "specific_schemes_json",
        "key_red_flags",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
