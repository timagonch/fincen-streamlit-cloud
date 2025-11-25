#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_summary_generator.py (v2 – canonical fraud families)

Generate LLM-based summaries + fraud-family metadata for FinCEN publications.

Pipeline (Supabase-native, doc_key-based):

INPUT TABLES
------------
- public.fincen_publications
    * doc_key (UNIQUE, GENERATED from title|doc_type|date)
    * title
    * doc_type
    * date
    * pdf_filename, pdf_url, detail_url, ...

- public.fincen_fulltext
    * doc_key (FK → fincen_publications.doc_key)
    * title
    * doc_type
    * date
    * full_text

OUTPUT TABLE
------------
- public.fincen_llm_summaries
    * doc_key                 (FK → fincen_publications.doc_key, UNIQUE)
    * title
    * doc_type
    * date
    * high_level_summary      (TEXT)
    * primary_fraud_families   (JSONB: list[str] – CANONICAL IDs ONLY)
    * secondary_fraud_families (JSONB: list[str] – CANONICAL IDs ONLY)
    * specific_schemes         (JSONB: list[{"fraud_family","scheme_label","description"}])
                                where fraud_family is a canonical ID
    * key_red_flags            (JSONB: list[str])

Summary generation is idempotent:
  - By default, only documents whose doc_key is NOT in fincen_llm_summaries
    are processed.
  - Use --force to re-summarize everything.

Usage:
  uv run fincen_summary_generator.py
  uv run fincen_summary_generator.py --limit 10
  uv run fincen_summary_generator.py --force

Environment:
  - GEMINI_API_KEY must be set in your .env or shell.
  - GEMINI_MODEL (optional) defaults to "gemini-2.5-flash".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai

from supabase_helpers import get_supabase_client


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_CHARS = 20000  # hard cap on characters sent to LLM per document


def _require_env_var(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(
            f"Environment variable {name} is not set. "
            f"Set {name} in your environment or .env file."
        )
    return val


def get_gemini_client() -> genai.Client:
    """
    Create a Google GenAI client using the GEMINI_API_KEY env var.
    """
    api_key = _require_env_var("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client


@dataclass
class DocRecord:
    doc_key: str
    title: str
    doc_type: Optional[str]
    date: Optional[str]
    full_text: str


# ---------------------------------------------------------------------------
# Canonical fraud families
# ---------------------------------------------------------------------------

# These are the ONLY allowed IDs the model should use in
# primary_fraud_families / secondary_fraud_families / specific_schemes[*].fraud_family.
#
# The IDs are what we store in the DB and use for charts.
# The labels/descriptions are just to guide the LLM.
CANONICAL_FRAUD_FAMILIES: List[Dict[str, str]] = [
    {
        "id": "money_laundering",
        "label": "Money laundering",
        "description": (
            "Movement, layering, or integration of illicit proceeds through the "
            "financial system (including use of shell companies, front businesses, "
            "bulk cash, or complex account structures)."
        ),
    },
    {
        "id": "terrorist_financing",
        "label": "Terrorist financing",
        "description": (
            "Raising, moving, or storing funds to support terrorist organizations, "
            "networks, or acts of terrorism."
        ),
    },
    {
        "id": "sanctions_evasion_proliferation",
        "label": "Sanctions evasion & proliferation",
        "description": (
            "Evasion of U.S. or international sanctions, or financing related to "
            "weapons proliferation, WMD programs, or restricted jurisdictions."
        ),
    },
    {
        "id": "fraud_scams_individuals",
        "label": "Fraud & scams targeting individuals",
        "description": (
            "Consumer-focused fraud and scams such as romance scams, elder "
            "financial exploitation, tech-support scams, lottery/sweepstakes scams, "
            "and similar schemes against individuals."
        ),
    },
    {
        "id": "fraud_scams_businesses",
        "label": "Fraud & scams targeting businesses",
        "description": (
            "Fraud schemes primarily targeting businesses or organizations, such as "
            "business email compromise (BEC), invoice fraud, vendor impersonation, "
            "procurement fraud, or corporate account abuse."
        ),
    },
    {
        "id": "cybercrime",
        "label": "Cybercrime",
        "description": (
            "Criminal activity that relies on computer systems or networks, such as "
            "ransomware, hacking, malware, credential theft, or large-scale data "
            "breaches, including the related movement of proceeds."
        ),
    },
    {
        "id": "corruption_bribery",
        "label": "Corruption & bribery",
        "description": (
            "Bribery of public or private officials, embezzlement, misuse of "
            "public funds, or other corruption-related abuse of position."
        ),
    },
    {
        "id": "trade_based_money_laundering",
        "label": "Trade-based money laundering (TBML)",
        "description": (
            "Use of trade transactions to move or disguise value, such as "
            "over/under-invoicing, phantom shipments, or misclassification of "
            "goods and services."
        ),
    },
    {
        "id": "human_trafficking_smuggling",
        "label": "Human trafficking & smuggling",
        "description": (
            "Financial activity related to human trafficking, sexual exploitation, "
            "forced labor, or migrant smuggling networks."
        ),
    },
    {
        "id": "identity_theft_account_takeover",
        "label": "Identity theft & account takeover",
        "description": (
            "Impersonation of customers or beneficial owners, use of stolen or "
            "synthetic identities, or takeover of existing accounts to move funds."
        ),
    },
    {
        "id": "tax_evasion_tax_crimes",
        "label": "Tax evasion & tax crimes",
        "description": (
            "Criminal tax evasion, willful under-reporting, use of structures or "
            "schemes primarily to evade tax obligations."
        ),
    },
    {
        "id": "illicit_drug_trafficking",
        "label": "Illicit drug trafficking",
        "description": (
            "Production, distribution, or financing of illegal narcotics or "
            "controlled substances, including the laundering of drug proceeds."
        ),
    },
    {
        "id": "other",
        "label": "Other / not clearly specified",
        "description": (
            "Use ONLY when the primary activity does not reasonably fit any other "
            "fraud family above."
        ),
    },
]


def _canonical_families_block() -> str:
    """
    Render the canonical families as a concise text block for the prompt.
    """
    lines = []
    for fam in CANONICAL_FRAUD_FAMILIES:
        lines.append(
            f'- "{fam["id"]}": {fam["label"]} – {fam["description"]}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Supabase loading
# ---------------------------------------------------------------------------


def load_docs_to_summarize(
    limit: Optional[int] = None, force: bool = False
) -> List[DocRecord]:
    """
    Load documents from fincen_fulltext, and filter to those that do not yet
    have summaries in fincen_llm_summaries (unless force=True).
    """
    client = get_supabase_client()

    print("[*] Loading full-text documents from Supabase…")
    full_resp = (
        client.table("fincen_fulltext")
        .select("doc_key, title, doc_type, date, full_text")
        .execute()
    )
    full_rows: List[Dict[str, Any]] = full_resp.data or []
    if not full_rows:
        print("[!] No rows found in fincen_fulltext. Have you run the text extractor?")
        return []

    print(f"    Loaded {len(full_rows)} rows from fincen_fulltext.")

    existing_keys: set[str] = set()
    if not force:
        print("[*] Loading existing summaries from fincen_llm_summaries…")
        summ_resp = client.table("fincen_llm_summaries").select("doc_key").execute()
        summ_rows = summ_resp.data or []
        existing_keys = {
            str(row["doc_key"])
            for row in summ_rows
            if row.get("doc_key") is not None
        }
        print(f"    Found {len(existing_keys)} doc_keys already summarized.")

    docs: List[DocRecord] = []
    for row in full_rows:
        doc_key = str(row.get("doc_key") or "").strip()
        if not doc_key:
            continue

        if not force and doc_key in existing_keys:
            continue

        title = str(row.get("title") or "").strip()
        doc_type = row.get("doc_type")
        date = row.get("date")
        full_text = row.get("full_text") or ""
        if not full_text.strip():
            # Nothing to summarize
            continue

        docs.append(
            DocRecord(
                doc_key=doc_key,
                title=title,
                doc_type=str(doc_type) if doc_type is not None else None,
                date=str(date) if date is not None else None,
                full_text=str(full_text),
            )
        )

    if limit is not None:
        docs = docs[:limit]

    print(f"[*] Will generate summaries for {len(docs)} documents.")
    return docs


# ---------------------------------------------------------------------------
# Prompt + LLM call
# ---------------------------------------------------------------------------


def _truncate_text_for_llm(full_text: str) -> str:
    """
    Truncate very long documents so we stay within a reasonable token budget.

    Strategy:
      - If under MAX_CHARS, return as-is.
      - Otherwise, keep the first ~60% and last ~40% of the text,
        joined with a clear '[TRUNCATED]' marker.
    """
    if len(full_text) <= MAX_CHARS:
        return full_text

    head_chars = int(MAX_CHARS * 0.6)
    tail_chars = MAX_CHARS - head_chars
    head = full_text[:head_chars]
    tail = full_text[-tail_chars:]
    return head + "\n\n[... TRUNCATED ...]\n\n" + tail


def build_prompt(meta: DocRecord, text: str) -> str:
    """
    Build a concise, structured prompt that asks Gemini to behave like a
    FinCEN-focused fraud analyst and return STRICT JSON with:

      - high_level_summary (2–4 sentences)
      - primary_fraud_families (1–3 CANONICAL IDs)
      - secondary_fraud_families (0–4 CANONICAL IDs)
      - specific_schemes (up to ~6 entries)
      - key_red_flags (up to ~8 entries)

    All fraud families MUST use canonical IDs from CANONICAL_FRAUD_FAMILIES.
    """
    doc_header = f"{meta.title or 'Unknown Title'}"
    if meta.doc_type:
        doc_header += f" ({meta.doc_type})"
    if meta.date:
        doc_header += f", {meta.date}"

    canonical_block = _canonical_families_block()

    return f"""
You are a senior financial crime analyst specializing in FinCEN advisories, alerts, and notices.

You will be given the FULL TEXT (or a slightly truncated version) of a single FinCEN publication.
Your job is to identify the main financial crime issues, fraud families, typologies, and red flags.

You MUST obey the following rules for FRAUD FAMILIES:

1. You have a fixed list of canonical fraud families. Each has an `id` that we store in a database:

{canonical_block}

2. In your JSON output:
   - `primary_fraud_families` MUST be a list of 1–3 of these canonical **id** strings,
     chosen from the list above. Do NOT invent new names.
   - `secondary_fraud_families` MUST also be drawn ONLY from these canonical ids.
   - For each object in `specific_schemes`, the `fraud_family` field MUST be one of
     these canonical ids as well.
   - If the activity in the document does not clearly fit any family, assign it to `"other"`.
3. You may NOT create new fraud family names or synonyms. Use ONLY the `id` values shown above.

Return your answer in STRICT JSON with the following keys:

{{
  "high_level_summary": "2–4 sentences in plain English. Focus on the main financial crime problem, who is involved, and why FinCEN cares.",

  "primary_fraud_families": [
    "Canonical fraud family ids for the main focus of this publication (e.g. 'money_laundering', 'fraud_scams_individuals'). Use 1–3 values."
  ],

  "secondary_fraud_families": [
    "Other relevant but less central fraud families, also as canonical ids. May be empty."
  ],

  "specific_schemes": [
    {{
      "fraud_family": "One canonical fraud family id from the list above.",
      "scheme_label": "Very short label for the specific scheme or typology (string).",
      "description": "1–2 sentence description of how this scheme works in this publication."
    }}
  ],

  "key_red_flags": [
    "Short bullet-style red flag indicators mentioned or implied in the publication. Each should be concise and operational."
  ]
}}

ADDITIONAL GUIDELINES:
- Keep all lists reasonably short (no more than about 6–8 items per list).
- Use consistent canonical fraud family ids across `primary_fraud_families`,
  `secondary_fraud_families`, and `specific_schemes[*].fraud_family`.
- If you are unsure, make your best professional judgment rather than returning empty lists,
  but still choose from the canonical ids above.
- Output MUST be valid JSON. Do NOT include any additional commentary, markdown, or explanation.

Document metadata:
- Title / ID: {doc_header}

--- BEGIN DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---
""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from the model's text response.
    Handles cases where the model wraps JSON in markdown fences like ```json.
    """
    if not text:
        raise ValueError("Empty LLM response")

    # Strip markdown code fences if present
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != -1 and end != -1 and end > start:
            inner = text[start + 3 : end]
            inner = inner.strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            text = inner.strip()

    # Fallback: try to grab from first '{' to last '}'
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        snippet = text[start : end + 1]
    else:
        snippet = text

    return json.loads(snippet)


def call_gemini_for_doc(
    client: genai.Client, meta: DocRecord
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini for a single document, returning a parsed JSON dict or None on failure.
    """
    prompt_text = _truncate_text_for_llm(meta.full_text)
    prompt = build_prompt(meta, prompt_text)

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[!] Gemini API error for {meta.doc_key}: {e}", file=sys.stderr)
        return None

    try:
        raw_text = response.text or ""
    except AttributeError:
        raw_text = str(response)

    try:
        payload = _extract_json(raw_text)
    except Exception as e:  # noqa: BLE001
        print(
            f"[!] Failed to parse JSON for {meta.doc_key}: {e}. Raw response snippet:\n"
            f"{raw_text[:500]}...\n",
            file=sys.stderr,
        )
        return None

    return payload


# ---------------------------------------------------------------------------
# Supabase write-back
# ---------------------------------------------------------------------------


def upsert_summary(
    supabase_client,
    meta: DocRecord,
    payload: Dict[str, Any],
) -> None:
    """
    Upsert a single document's summary into fincen_llm_summaries.
    """
    record: Dict[str, Any] = {
        "doc_key": meta.doc_key,
        "title": meta.title,
        "doc_type": meta.doc_type,
        "date": meta.date,
        "high_level_summary": (payload.get("high_level_summary") or "").strip(),
        "primary_fraud_families": payload.get("primary_fraud_families") or [],
        "secondary_fraud_families": payload.get("secondary_fraud_families") or [],
        "specific_schemes": payload.get("specific_schemes") or [],
        "key_red_flags": payload.get("key_red_flags") or [],
    }

    # Normalize None → []
    for key in [
        "primary_fraud_families",
        "secondary_fraud_families",
        "specific_schemes",
        "key_red_flags",
    ]:
        val = record[key]
        if val is None:
            record[key] = []

    supabase_client.table("fincen_llm_summaries").upsert(record).execute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(force: bool = False, limit: Optional[int] = None) -> None:
    docs = load_docs_to_summarize(limit=limit, force=force)
    if not docs:
        print("[*] Nothing to summarize. Exiting.")
        return

    gemini_client = get_gemini_client()
    supabase_client = get_supabase_client()

    processed = 0
    for meta in docs:
        print(
            f"\n[*] [{processed + 1}/{len(docs)}] Summarizing: "
            f"{meta.title or meta.doc_key}…"
        )

        payload = call_gemini_for_doc(gemini_client, meta)
        if not payload:
            print(f"[!] Skipping {meta.doc_key} due to errors.")
            continue

        upsert_summary(supabase_client, meta, payload)
        processed += 1
        # Gentle pacing to avoid hammering the API
        time.sleep(0.5)

    print(f"\n[*] Done. Successfully processed {processed} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM summaries for FinCEN publications using "
            "fincen_fulltext + fincen_llm_summaries (canonical fraud families)."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize even if fincen_llm_summaries already has a row for a document.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing).",
    )

    args = parser.parse_args()

    main(force=args.force, limit=args.limit)
