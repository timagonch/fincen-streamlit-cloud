#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_summary_generator.py
---------------------------------
Generate structured LLM summaries for each FinCEN publication using Gemini.

Inputs (expected in the same folder by default):
  - fincen_fraud_mapping.csv
  - fincen_semantic_chunks.csv

Output:
  - fincen_summaries.json

Each entry contains:
  - article_name, fincen_id, doc_type, doc_title, doc_date
  - high_level_summary
  - primary_fraud_types
  - key_red_flags
  - recommended_sar_focus
  - why_it_matters_for_usaa

This script is meant for offline batch summarization, not live calls inside Streamlit.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from google import genai
from google.genai import errors as genai_errors


# Load environment variables from .env in project root (including GEMINI_API_KEY)
load_dotenv()


# ------------------------ CONFIG (tweak as needed) ----------------------------

MAPPING_CSV = "fincen_fraud_mapping.csv"
CHUNKS_CSV = "fincen_semantic_chunks.csv"
OUT_JSON = "fincen_summaries.json"

# Gemini model – use a fast, cheaper one for batch summaries
GEMINI_MODEL = "gemini-2.5-flash"

# How many text chunks per document to feed into the LLM.
MAX_CHUNKS_PER_DOC = 12

# Max characters per chunk we will pass to the LLM (defensive)
MAX_CHARS_PER_CHUNK = 1200


# ------------------------------- Data models ----------------------------------


@dataclass
class DocSummary:
    article_name: str
    fincen_id: str
    doc_type: str
    doc_title: str
    doc_date: str

    high_level_summary: str
    primary_fraud_types: List[str]
    key_red_flags: List[str]
    recommended_sar_focus: List[str]
    why_it_matters_for_usaa: str


# ----------------------------- Helper functions -------------------------------


def _require_env_var(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(
            f"Environment variable {name} is not set.\n"
            "Get an API key from Google AI Studio and set it, e.g.:\n"
            "  GESMINI_API_KEY=your_key in .env or\n"
            "  setx GEMINI_API_KEY \"YOUR_KEY_HERE\""
        )
    return val


def load_inputs(
    mapping_path: Path = Path(MAPPING_CSV),
    chunks_path: Path = Path(CHUNKS_CSV),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping CSV: {mapping_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing semantic chunks CSV: {chunks_path}")

    mapping = pd.read_csv(mapping_path)
    chunks = pd.read_csv(chunks_path)

    # Normalize expected columns on mapping
    for col in ["article_name", "fincen_id", "doc_type", "doc_title", "doc_date", "top_labels_regex"]:
        if col not in mapping.columns:
            mapping[col] = ""

    # Normalize expected columns on chunks
    if "article_name" not in chunks.columns:
        # Fallback: if "file" exists, extract from that
        if "file" in chunks.columns:
            chunks["article_name"] = chunks["file"].astype(str).apply(
                lambda p: Path(p).name
            )
        else:
            chunks["article_name"] = ""

    if "matched_fraud_types" not in chunks.columns:
        chunks["matched_fraud_types"] = ""

    if "page_number" not in chunks.columns:
        chunks["page_number"] = 1

    if "text" not in chunks.columns:
        chunks["text"] = ""

    return mapping, chunks


def select_docs(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    For now, include all docs. You could add filters here
    (e.g., limit to recent_years or specific doc_type).
    """
    return mapping.copy()


def select_chunks_for_doc(
    chunks: pd.DataFrame, article_name: str
) -> List[str]:
    """
    Pick a limited set of the most useful chunks for a given article.
    Strategy:
      - Only chunks for this article
      - Prefer chunks that have any matched_fraud_types
      - Then fall back to other chunks if needed
      - Sort by page_number
    """
    subset = chunks[chunks["article_name"] == article_name].copy()
    if subset.empty:
        return []

    subset["matched_fraud_types"] = (
        subset["matched_fraud_types"].fillna("").astype(str)
    )

    # Flag chunks that have at least one fraud label
    subset["has_label"] = subset["matched_fraud_types"].apply(
        lambda s: bool(s.strip())
    )

    # Sort: labeled chunks first, then by page
    subset = subset.sort_values(
        by=["has_label", "page_number"], ascending=[False, True]
    )

    chosen: List[str] = []
    for _, row in subset.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        # Hard cap length per chunk
        if len(text) > MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK].rstrip() + " …"
        chosen.append(text)
        if len(chosen) >= MAX_CHUNKS_PER_DOC:
            break

    return chosen


def build_prompt(
    meta: Dict[str, str], chunks: List[str], fraud_label_string: str
) -> str:
    """
    Build a single text prompt for Gemini to summarize a publication.
    We keep this as a pure-text prompt for simplicity.
    """
    header = f"""
You are a senior financial crime analyst preparing a briefing for a large US financial institution's fraud team (similar to USAA).

You are summarizing a single FinCEN publication. Your audience are AML investigators, FIU analysts, and fraud strategy leaders. Be concise, specific, and practical.

Publication metadata:
- FinCEN ID: {meta.get('fincen_id') or 'N/A'}
- Document type: {meta.get('doc_type') or 'N/A'}
- Title: {meta.get('doc_title') or 'N/A'}
- Date: {meta.get('doc_date') or 'N/A'}

Heuristic fraud labels from our NLP pipeline (regex counts per type):
{fraud_label_string or 'None detected'}

Below are extracted paragraphs and sentences from the PDF (not the full text), in reading order. Some may be truncated:

-------------------- START EXTRACTED TEXT --------------------
"""
    body = "\n\n".join(chunks)
    footer = """
-------------------- END EXTRACTED TEXT --------------------

Using ONLY the information above, produce a JSON object with the following fields:

{
  "high_level_summary": "<2–4 sentence plain-language summary of what this publication is about and what FinCEN is asking institutions to do>",
  "primary_fraud_types": [
    "<short fraud type label 1>",
    "<short fraud type label 2>"
  ],
  "key_red_flags": [
    "<bullet-style red flag / typology indicator 1>",
    "<bullet-style red flag / typology indicator 2>"
  ],
  "recommended_sar_focus": [
    "<how SAR narratives or monitoring rules should mention this guidance>",
    "<specific products / channels / counterparties to pay attention to>"
  ],
  "why_it_matters_for_usaa": "<brief paragraph explaining how this advisory/alert/notice would change monitoring priorities, risk assessments, or investigator training for a large US retail FI like USAA>"
}
"""
    footer = """
-------------------- END EXTRACTED TEXT --------------------

Using ONLY the information above, produce a JSON object with the following fields:

{
  "high_level_summary": "<2–4 sentence plain-language summary of what this publication is about and what FinCEN is asking institutions to do>",
  "primary_fraud_types": [
    "<short fraud type label 1>",
    "<short fraud type label 2>"
  ],
  "key_red_flags": [
    "<bullet-style red flag / typology indicator 1>",
    "<bullet-style red flag / typology indicator 2>"
  ],
  "recommended_sar_focus": [
    "<how SAR narratives or monitoring rules should mention this guidance>",
    "<specific products / channels / counterparties to pay attention to>"
  ],
  "why_it_matters_for_usaa": "<brief paragraph explaining how this advisory/alert/notice would change monitoring priorities, risk assessments, or investigator training for a large US retail FI like USAA>"
}

Important:
- Respond with STRICTLY valid JSON.
- Do NOT wrap the JSON in markdown code fences like ```json or ```anything.
- Do NOT include any explanation outside the JSON.
- If information isn't available, use an empty string or empty list.
"""

    return header + body + footer


def make_gemini_client() -> genai.Client:
    api_key = _require_env_var("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client


def load_existing_summaries(path: Path) -> Dict[str, Dict]:
    """Load existing summaries JSON if present; otherwise return empty dict."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If file is corrupted or not valid JSON, ignore it and start fresh
        return {}

def _clean_str(val: object) -> str:
    """Convert NaN/None/'nan' to a clean empty string-safe value."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def summarize_one_doc(
    client: genai.Client,
    meta_row: pd.Series,
    chunks_for_doc: List[str],
) -> Optional[DocSummary]:
    """Call Gemini for a single document and parse the JSON result."""
    if not chunks_for_doc:
        # Nothing to summarize
        return None

    fraud_label_string = str(meta_row.get("top_labels_regex", "") or "").strip()

    meta = {
        "article_name": _clean_str(meta_row.get("article_name", "")),
        "fincen_id": _clean_str(meta_row.get("fincen_id", "")),
        "doc_type": _clean_str(meta_row.get("doc_type", "")),
        "doc_title": _clean_str(meta_row.get("doc_title", "")),
        "doc_date": _clean_str(meta_row.get("doc_date", "")),
    }

    prompt = build_prompt(meta, chunks_for_doc, fraud_label_string)

    # --- Robust Gemini call with retries on 503/overload ---
    max_attempts = 3
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            break  # success, leave the retry loop
        except genai_errors.ServerError as e:
            # 5xx: model overloaded / transient
            print(f"  -> Gemini server error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(5)  # brief backoff, then retry
            else:
                print("  -> Giving up on this document for now.")
                return None
        except Exception as e:
            # Anything else: log and skip this doc, but don't crash the batch
            print(f"  -> Unexpected Gemini error: {e}")
            return None

    if response is None:
        return None

    raw_text = (response.text or "").strip()
    if not raw_text:
        return None

    # --- Strip markdown code fences if Gemini ignored instructions ---
    text = raw_text
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line if it starts with ``` (with or without language)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop last line if it's a closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to parse JSON; if it fails, store raw text as high_level_summary
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("  -> Warning: model did not return valid JSON; storing raw text.")
        return DocSummary(
            article_name=meta["article_name"],
            fincen_id=meta["fincen_id"],
            doc_type=meta["doc_type"],
            doc_title=meta["doc_title"],
            doc_date=meta["doc_date"],
            high_level_summary=raw_text,
            primary_fraud_types=[],
            key_red_flags=[],
            recommended_sar_focus=[],
            why_it_matters_for_usaa="",
        )

    def _get_list(key: str) -> List[str]:
        val = data.get(key, [])
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    return DocSummary(
        article_name=meta["article_name"],
        fincen_id=meta["fincen_id"],
        doc_type=meta["doc_type"],
        doc_title=meta["doc_title"],
        doc_date=meta["doc_date"],
        high_level_summary=_clean_str(data.get("high_level_summary", "")),
        primary_fraud_types=_get_list("primary_fraud_types"),
        key_red_flags=_get_list("key_red_flags"),
        recommended_sar_focus=_get_list("recommended_sar_focus"),
        why_it_matters_for_usaa=_clean_str(
            data.get("why_it_matters_for_usaa", "")
        ),
    )



# ------------------------------- Main -----------------------------------------


def main(force: bool = False) -> None:
    base = Path(".")
    print("[*] Loading inputs…")
    mapping, chunks = load_inputs(base / MAPPING_CSV, base / CHUNKS_CSV)

    docs = select_docs(mapping)
    if docs.empty:
        print("No documents found in mapping CSV.")
        return

    out_path = base / OUT_JSON
    existing = load_existing_summaries(out_path)
    print(f"[*] Existing summaries loaded: {len(existing)}")

    client = make_gemini_client()

    summaries: Dict[str, Dict] = dict(existing)  # start from existing
    total = len(docs)

    for idx, (_, row) in enumerate(docs.iterrows(), start=1):
        article_name = str(row.get("article_name", "") or "")

        if not force and article_name in summaries:
            print(f"[{idx}/{total}] {article_name} -> already summarized, skipping.")
            continue

        print(f"[{idx}/{total}] Summarizing: {article_name} …")

        doc_chunks = select_chunks_for_doc(chunks, article_name)
        if not doc_chunks:
            print("  -> No chunks found for this document; skipping.")
            continue

        summary_obj = summarize_one_doc(client, row, doc_chunks)
        if summary_obj is None:
            print("  -> No summary generated; skipping.")
            continue

        summaries[article_name] = asdict(summary_obj)

        # Flush to disk after each new summary so a crash doesn't lose progress
        out_path.write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  -> Saved summary for {article_name}")

    print(f"\nDone. Wrote {len(summaries)} total summaries to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LLM summaries for FinCEN publications using Gemini."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize all documents, even if they already have summaries.",
    )
    args = parser.parse_args()

    main(force=args.force)
