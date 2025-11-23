#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_summary_generator.py (lean version)

Ruthless, cost-conscious version.

- Uses Gemini Flash with a small, tight prompt.
- Only sends a handful of high-value chunks per document (max 8).
- Outputs only the fields we actually need for the dashboard & analysis.

Inputs:
  - fincen_fraud_mapping.csv      (document-level semantic metadata)
  - fincen_semantic_chunks.csv    (chunk-level semantic tagging)

Output:
  - fincen_summaries.json  (dict keyed by article_name)

Each entry:
  - article_name: PDF filename (stable technical ID)
  - fincen_id: FinCEN identifier if available
  - doc_type: Advisory | Alert | Notice | other
  - doc_title: Human-readable title from crawler metadata (or inferred upstream)
  - doc_date: Publication date (string)

  - high_level_summary: 2–4 sentence summary
  - primary_fraud_families: 1–3 keys from FRAUD_TYPES
  - secondary_fraud_families: optional additional FRAUD_TYPES keys
  - specific_schemes: list of {family, label, notes}
  - key_red_flags: list of bullet-style strings

This script is incremental:
  - If fincen_summaries.json exists, we load it and skip any article_name
    that already has an entry, unless --force is used.

We deliberately *do not* compute semantic_fraud_families here; that can be
derived cheaply later from fincen_fraud_mapping.csv & fincen_semantic_chunks.csv
without extra LLM calls.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

MAPPING_CSV = "fincen_fraud_mapping.csv"
CHUNKS_CSV = "fincen_semantic_chunks.csv"
OUT_JSON = "fincen_summaries.json"

GEMINI_MODEL = "gemini-2.5-flash"

# Ruthless token control:
MAX_CHUNKS_PER_DOC = 8
MAX_CHARS_PER_CHUNK = 900  # keep chunks compact

# Fraud family ontology (must stay in sync with fincen_ocr_fraud_mapper.py)
from typing import Dict as _Dict

FRAUD_TYPES: _Dict[str, str] = {
    "money_laundering": "Money laundering and layering of criminal proceeds through financial institutions.",
    "structuring_smurfing": "Structuring or smurfing to avoid reporting thresholds, such as CTRs.",
    "terrorist_financing": "Financing of terrorism, extremist organizations, or designated persons.",
    "sanctions_evasion": "Sanctions evasion and dealing with OFAC-designated jurisdictions or parties.",
    "human_trafficking": "Human trafficking or smuggling, exploitation of workers or victims for profit.",
    "human_smuggling": "Human smuggling across borders, paying facilitators to move people illegally.",
    "fraud_government_programs": "Fraud against government programs such as unemployment insurance, EIP, PPP, healthcare benefits.",
    "ransomware_cyber": "Ransomware, cyber extortion, or related cyber-enabled financial crime.",
    "trade_based_money_laundering": "Trade-based money laundering and abuse of invoices, shipping, import and export flows.",
    "bulk_cash_smuggling": "Bulk cash smuggling, cross-border movement of large amounts of currency.",
    "virtual_assets_crypto": "Use of virtual assets or cryptocurrencies such as Bitcoin or stablecoins for illicit finance.",
    "correspondent_banking": "Misuse of correspondent banking relationships and nested accounts.",
    "shell_front_companies": "Use of shell companies or front businesses to hide ownership or illicit proceeds.",
    "elder_romance_scam": "Elder fraud, romance scams, or schemes targeting vulnerable customers.",
    "synthetic_identity": "Synthetic identity creation and misuse of identity information.",
    "proliferation_financing": "Financing of proliferation of weapons of mass destruction.",
    "corruption_bribery": "Corruption, bribery, embezzlement, misuse of public office.",
}

FRAUD_FAMILY_HELP = "\n".join(
    f"- {key}: {desc}" for key, desc in FRAUD_TYPES.items()
)


@dataclass
class DocSummary:
    article_name: str
    fincen_id: str
    doc_type: str
    doc_title: str
    doc_date: str

    high_level_summary: str
    primary_fraud_families: List[str]
    secondary_fraud_families: List[str]
    specific_schemes: List[Dict[str, str]]
    key_red_flags: List[str]


def _require_env_var(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(
            f"Environment variable {name} is not set. "
            "Set GEMINI_API_KEY in your environment or .env file."
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

    # Ensure minimal columns exist
    for col in ["article_name", "fincen_id", "doc_type", "doc_title", "doc_date"]:
        if col not in mapping.columns:
            mapping[col] = ""

    for col in ["article_name", "text", "is_high_signal", "chunk_role", "page_number"]:
        if col not in chunks.columns:
            if col == "is_high_signal":
                chunks[col] = False
            else:
                chunks[col] = ""

    return mapping, chunks


def select_docs(mapping: pd.DataFrame) -> pd.DataFrame:
    # For now, include everything; easy to filter later (by year, doc_type, etc.)
    return mapping.copy()


def select_chunks_for_doc(chunks: pd.DataFrame, article_name: str) -> List[str]:
    """Pick a small, high-value subset of chunks for this document.

    Ruthless strategy:
      1. Always include the first page's text (title + intro signal).
      2. Then add high-signal chunks (typology / red_flag / sar_guidance / regulatory).
      3. Cap at MAX_CHUNKS_PER_DOC and MAX_CHARS_PER_CHUNK.

    We do *not* try to be perfect; just enough signal for doc-level summary.
    """
    doc_chunks = chunks[chunks["article_name"] == article_name].copy()
    if doc_chunks.empty:
        return []

    # Ensure types
    doc_chunks["is_high_signal"] = doc_chunks["is_high_signal"].fillna(False).astype(bool)
    if "page_number" in doc_chunks.columns:
        doc_chunks["page_number"] = pd.to_numeric(doc_chunks["page_number"], errors="coerce").fillna(0).astype(int)
    else:
        doc_chunks["page_number"] = 0

    # 1) First page chunks (cheap way to capture title/intro)
    first_page = doc_chunks[doc_chunks["page_number"] == doc_chunks["page_number"].min()].copy()

    # 2) High-signal chunks
    high = doc_chunks[doc_chunks["is_high_signal"]].copy()

    # Prioritize by semantic role
    def sort_by_role(df: pd.DataFrame) -> pd.DataFrame:
        priority = {
            "typology_description": 0,
            "red_flag_indicator": 1,
            "sar_guidance": 2,
            "regulatory_requirement": 3,
        }
        df = df.copy()
        df["role_rank"] = df["chunk_role"].map(priority).fillna(10)
        return df.sort_values(["role_rank", "page_number"]).drop(columns=["role_rank"])

    first_page = sort_by_role(first_page)
    high = sort_by_role(high)

    chosen: List[str] = []
    seen_idx = set()

    def add_from(df: pd.DataFrame):
        nonlocal chosen, seen_idx
        for idx, row in df.iterrows():
            if idx in seen_idx:
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if len(text) > MAX_CHARS_PER_CHUNK:
                text = text[:MAX_CHARS_PER_CHUNK].rstrip() + " …"
            chosen.append(text)
            seen_idx.add(idx)
            if len(chosen) >= MAX_CHUNKS_PER_DOC:
                break

    # First page first, then high-signal elsewhere
    add_from(first_page)
    if len(chosen) < MAX_CHUNKS_PER_DOC:
        add_from(high)

    return chosen


def build_prompt(meta: Dict[str, str], chunks: List[str]) -> str:
    fraud_family_block = FRAUD_FAMILY_HELP

    header = f"""You are a financial crime analyst summarizing a single FinCEN publication.

Publication metadata:
- FinCEN ID: {meta.get('fincen_id') or 'N/A'}
- Document type: {meta.get('doc_type') or 'N/A'}
- Existing title (may be empty): {meta.get('doc_title') or ''}

You will receive a small set of important text excerpts from the publication.

Your tasks:
1) Write a concise high_level_summary (2–4 sentences).
2) Classify the main fraud themes into 1–3 primary_fraud_families.
3) Optionally add any secondary_fraud_families that are clearly present.
4) For each chosen fraud family, list 1–3 short specific_schemes with:
   - family: the fraud family key
   - label: a short scheme name
   - notes: optional extra detail (can be empty)
5) Extract 3–10 key_red_flags that a bank could use to detect this activity.
6) If the existing title is empty or clearly generic, you may infer a better
   doc_title and use it in your reasoning, but DO NOT return it in JSON; the
   title field is handled upstream.

Allowed fraud family keys and their meanings:
{fraud_family_block}

Return JSON ONLY, no commentary, using this exact schema:

{{
  "high_level_summary": "<string>",
  "primary_fraud_families": ["<fraud_family_key>", "..."],
  "secondary_fraud_families": ["<fraud_family_key>", "..."],
  "specific_schemes": [
    {{
      "family": "<fraud_family_key>",
      "label": "<short scheme label>",
      "notes": "<optional extra detail>"
    }}
  ],
  "key_red_flags": [
    "<bullet-style red flag 1>",
    "<bullet-style red flag 2>"
  ]
}}

Text excerpts:
"""

    chunk_block_lines = []
    for i, ch in enumerate(chunks, start=1):
        chunk_block_lines.append(f"\n--- EXCERPT {i} ---\n{ch}")
    return header + "".join(chunk_block_lines)


def make_gemini_client() -> genai.Client:
    _require_env_var("GEMINI_API_KEY")
    return genai.Client()


def load_existing_summaries(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out: Dict[str, Dict[str, Any]] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            key = str(item.get("article_name") or f"row_{i}")
            out[key] = item
        return out
    return {}


def save_summaries(path: Path, summaries: Dict[str, Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def call_gemini_with_retry(
    client: genai.Client, prompt: str, max_retries: int = 3
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = resp.text or ""
            return extract_json_from_text(text)
        except (genai_errors.ClientError, genai_errors.ServerError, json.JSONDecodeError) as e:
            last_err = e
            print(f"[!] Gemini error {attempt}/{max_retries}: {e}")
            time.sleep(2 * attempt)
        except Exception as e:
            last_err = e
            print(f"[!] Unexpected error {attempt}/{max_retries}: {e}")
            time.sleep(2 * attempt)
    if last_err:
        raise last_err
    raise RuntimeError("Gemini call failed with no exception captured.")


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def summarize_single_doc(
    client: genai.Client,
    row: pd.Series,
    chunks_df: pd.DataFrame,
    existing: Dict[str, Dict[str, Any]],
    force: bool = False,
) -> Optional[DocSummary]:
    article_name = str(row.get("article_name", "")).strip()
    if not article_name:
        print("[!] Skipping row with empty article_name.")
        return None

    if not force and article_name in existing:
        return None

    meta = {
        "article_name": article_name,
        "fincen_id": str(row.get("fincen_id") or "").strip(),
        "doc_type": str(row.get("doc_type") or "").strip(),
        "doc_title": str(row.get("doc_title") or "").strip(),
        "doc_date": str(row.get("doc_date") or "").strip(),
    }

    doc_chunks = select_chunks_for_doc(chunks_df, article_name)
    if not doc_chunks:
        print(f"[!] No chunks found for {article_name}; skipping.")
        return None

    prompt = build_prompt(meta, doc_chunks)
    llm_json = call_gemini_with_retry(client, prompt)

    high_level_summary = str(llm_json.get("high_level_summary", "") or "").strip()

    primary_families = [
        f for f in _normalize_str_list(llm_json.get("primary_fraud_families"))
        if f in FRAUD_TYPES
    ]
    secondary_families = [
        f for f in _normalize_str_list(llm_json.get("secondary_fraud_families"))
        if f in FRAUD_TYPES and f not in primary_families
    ]
    key_red_flags = _normalize_str_list(llm_json.get("key_red_flags"))

    raw_schemes = llm_json.get("specific_schemes") or []
    cleaned_schemes: List[Dict[str, str]] = []
    if isinstance(raw_schemes, list):
        for item in raw_schemes:
            if isinstance(item, dict):
                fam = str(item.get("family") or "").strip()
                label = str(item.get("label") or "").strip()
                notes = str(item.get("notes") or "").strip()
                if fam and fam not in FRAUD_TYPES:
                    fam = ""
                cleaned_schemes.append(
                    {"family": fam, "label": label, "notes": notes}
                )
            elif isinstance(item, str):
                label = item.strip()
                if label:
                    cleaned_schemes.append(
                        {"family": "", "label": label, "notes": ""}
                    )

    summary = DocSummary(
        article_name=meta["article_name"],
        fincen_id=meta["fincen_id"],
        doc_type=meta["doc_type"],
        doc_title=meta["doc_title"],
        doc_date=meta["doc_date"],
        high_level_summary=high_level_summary,
        primary_fraud_families=primary_families,
        secondary_fraud_families=secondary_families,
        specific_schemes=cleaned_schemes,
        key_red_flags=key_red_flags,
    )
    return summary


def main(force: bool = False, limit: Optional[int] = None, out_filename: Optional[str] = None) -> None:
    base = Path(".")
    print("[*] Loading mapping + chunks…")
    mapping, chunks = load_inputs(base / MAPPING_CSV, base / CHUNKS_CSV)

    docs = select_docs(mapping)
    if docs.empty:
        print("No documents found in mapping CSV.")
        return

    if limit is not None:
        docs = docs.head(limit)
        print(f"[*] Limiting to first {len(docs)} documents for this run.")

    # Allow overriding output filename (handy for test runs)
    out_name = out_filename or OUT_JSON
    out_path = base / out_name

    existing = load_existing_summaries(out_path)
    print(f"[*] Existing summaries in {out_name}: {len(existing)}")

    client = make_gemini_client()

    summaries: Dict[str, Dict[str, Any]] = dict(existing)

    total = len(docs)
    for idx, (_, row) in enumerate(docs.iterrows(), start=1):
        article_name = str(row.get("article_name") or "").strip()
        print(f"[*] [{idx}/{total}] {article_name!r}…")

        summary = summarize_single_doc(
            client, row, chunks_df=chunks, existing=summaries, force=force
        )
        if summary is None:
            continue

        summaries[summary.article_name] = asdict(summary)

        if idx % 5 == 0:
            print("[*] Saving intermediate summaries…")
            save_summaries(out_path, summaries)

    print(f"[*] Writing final summaries JSON to {out_name}…")
    save_summaries(out_path, summaries)
    print(f"[*] Done. Wrote {len(summaries)} summaries to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate lean LLM summaries for FinCEN publications."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize even if this output file already has summaries for a document.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Output JSON filename (default: {OUT_JSON}).",
    )

    args = parser.parse_args()

    main(force=args.force, limit=args.limit, out_filename=args.out)

