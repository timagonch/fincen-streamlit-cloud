#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_build_fraud_dictionary.py

Build a global fraud-family / scheme dictionary from LLM summaries.

INPUT:
  - fincen_llm_summaries
      * doc_key
      * title
      * primary_fraud_families   (JSONB: list[str])
      * specific_schemes         (JSONB: list[{fraud_family, scheme_label, description}])
      * key_red_flags            (JSONB: list[str])

OUTPUT:
  - fincen_fraud_dictionary
      * fraud_family      TEXT NOT NULL
      * scheme_label      TEXT NOT NULL
      * description       TEXT
      * example_red_flags TEXT[]
      * example_doc_keys  TEXT[]
    UNIQUE (fraud_family, scheme_label)

Usage:
  uv run fincen_build_fraud_dictionary.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from supabase_helpers import get_supabase_client


@dataclass
class SchemeAgg:
    fraud_family: str
    scheme_label: str
    description: Optional[str] = None
    example_doc_keys: Set[str] = field(default_factory=set)
    example_red_flags: Set[str] = field(default_factory=set)


def _normalize_family(name: str) -> str:
    """
    Normalize fraud family names a bit so we don't get silly duplicates like
    'Money Laundering' vs 'money laundering '.
    """
    return " ".join(str(name).strip().split()).title()


def _normalize_scheme_label(name: str) -> str:
    return " ".join(str(name).strip().split())


def load_summaries() -> List[Dict[str, Any]]:
    client = get_supabase_client()
    print("[*] Loading summaries from fincen_llm_summaries…")

    resp = client.table("fincen_llm_summaries").select(
        "doc_key, title, primary_fraud_families, specific_schemes, key_red_flags"
    ).execute()

    rows: List[Dict[str, Any]] = resp.data or []
    print(f"    Loaded {len(rows)} summary rows.")
    return rows


def build_aggregates(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], SchemeAgg]:
    """
    Aggregate specific schemes across all documents into a dictionary keyed by
    (fraud_family, scheme_label).
    """
    agg: Dict[Tuple[str, str], SchemeAgg] = {}

    for row in rows:
        doc_key = str(row.get("doc_key") or "").strip()
        if not doc_key:
            continue

        primary_families = row.get("primary_fraud_families") or []
        specific_schemes = row.get("specific_schemes") or []
        key_red_flags = row.get("key_red_flags") or []

        # Supabase JSONB comes back as python lists already, but be defensive:
        if isinstance(primary_families, str):
            try:
                primary_families = json.loads(primary_families)
            except Exception:  # noqa: BLE001
                primary_families = []

        if isinstance(specific_schemes, str):
            try:
                specific_schemes = json.loads(specific_schemes)
            except Exception:  # noqa: BLE001
                specific_schemes = []

        if isinstance(key_red_flags, str):
            try:
                key_red_flags = json.loads(key_red_flags)
            except Exception:  # noqa: BLE001
                key_red_flags = []

        # Make a small set of red flags for this doc (we'll union later)
        doc_red_flags = set()
        for rf in key_red_flags or []:
            rf_str = " ".join(str(rf).strip().split())
            if rf_str:
                doc_red_flags.add(rf_str)
        # We'll later limit how many we keep per scheme.

        for scheme in specific_schemes or []:
            if not isinstance(scheme, dict):
                continue

            raw_family = scheme.get("fraud_family") or ""
            raw_label = scheme.get("scheme_label") or ""
            description = scheme.get("description") or ""

            if not raw_family and primary_families:
                # Fallback: use first primary family if scheme didn't specify one
                raw_family = primary_families[0]

            family = _normalize_family(raw_family) if raw_family else "Uncategorized"
            label = _normalize_scheme_label(raw_label or (description[:60] + "..."))

            if not label.strip():
                continue

            key = (family, label)

            if key not in agg:
                agg[key] = SchemeAgg(
                    fraud_family=family,
                    scheme_label=label,
                    description=description.strip() or None,
                    example_doc_keys={doc_key},
                    example_red_flags=set(doc_red_flags),
                )
            else:
                entry = agg[key]
                # Prefer to keep the first non-empty description
                if not entry.description and description.strip():
                    entry.description = description.strip()

                entry.example_doc_keys.add(doc_key)
                entry.example_red_flags.update(doc_red_flags)

    return agg


def upsert_dictionary(agg: Dict[Tuple[str, str], SchemeAgg]) -> None:
    client = get_supabase_client()

    print(f"[*] Upserting {len(agg)} entries into fincen_fraud_dictionary…")

    records: List[Dict[str, Any]] = []
    for key, entry in agg.items():
        # Limit example lists so they don't explode
        doc_keys_list = list(entry.example_doc_keys)
        red_flags_list = list(entry.example_red_flags)

        if len(doc_keys_list) > 10:
            doc_keys_list = doc_keys_list[:10]

        if len(red_flags_list) > 10:
            red_flags_list = red_flags_list[:10]

        records.append(
            {
                "fraud_family": entry.fraud_family,
                "scheme_label": entry.scheme_label,
                "description": entry.description or "",
                "example_doc_keys": doc_keys_list,
                "example_red_flags": red_flags_list,
            }
        )

    # Upsert in chunks to be gentle
    chunk_size = 100
    for i in range(0, len(records), chunk_size):
        batch = records[i : i + chunk_size]
        client.table("fincen_fraud_dictionary").upsert(batch).execute()
        print(f"    Upserted {i + len(batch)} / {len(records)}")

    print("[*] Done updating fincen_fraud_dictionary.")


def main() -> None:
    rows = load_summaries()
    if not rows:
        print("[*] No summaries found; run fincen_summary_generator.py first.")
        return

    agg = build_aggregates(rows)
    if not agg:
        print("[*] No schemes found in summaries; nothing to do.")
        return

    upsert_dictionary(agg)


if __name__ == "__main__":
    main()
