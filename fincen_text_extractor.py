#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_text_extractor.py

Step 1 in the new pipeline:
  - Take PDFs referenced in `fincen_publications`
  - Download them from Supabase Storage
  - Extract text (PyMuPDF + OCR fallback, reusing mapper code)
  - Store one `full_text` per doc into `fincen_fulltext`

INPUT:
  - fincen_publications
      * doc_key (generated)
      * title
      * doc_type
      * date
      * pdf_filename

  - Supabase storage bucket "fincen-pdfs"
      * advisories/<pdf_filename>
      * alerts/<pdf_filename>
      * notices/<pdf_filename>

OUTPUT:
  - fincen_fulltext
      * doc_key  (FK → fincen_publications.doc_key, UNIQUE)
      * title
      * doc_type
      * date
      * full_text

Usage:
  uv run fincen_text_extractor.py
  uv run fincen_text_extractor.py --limit 10
  uv run fincen_text_extractor.py --force  (re-extract everything)

"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from supabase_helpers import get_supabase_client
from fincen_ocr_fraud_mapper import (
    download_pdf_from_bucket,
    extract_chunks_from_pdf,
    Chunk,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PubRecord:
    doc_key: str
    title: str
    doc_type: Optional[str]
    date: Optional[str]
    pdf_filename: str


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def load_publications_to_extract(
    limit: Optional[int] = None,
    force: bool = False,
) -> List[PubRecord]:
    """
    Load publications from fincen_publications and filter to only those that
    do NOT yet have a row in fincen_fulltext (unless force=True).
    """
    client = get_supabase_client()

    print("[*] Loading publications from fincen_publications…")
    pubs_resp = (
        client.table("fincen_publications")
        .select("doc_key, title, doc_type, date, pdf_filename")
        .execute()
    )
    pubs_rows: List[Dict[str, Any]] = pubs_resp.data or []
    if not pubs_rows:
        print("[!] No rows in fincen_publications.")
        return []

    print(f"    Loaded {len(pubs_rows)} publication rows.")

    existing_keys: set[str] = set()
    if not force:
        print("[*] Loading existing full-text docs from fincen_fulltext…")
        full_resp = client.table("fincen_fulltext").select("doc_key").execute()
        full_rows = full_resp.data or []
        existing_keys = {
            str(row["doc_key"])
            for row in full_rows
            if row.get("doc_key") is not None
        }
        print(f"    Found {len(existing_keys)} doc_keys already in fincen_fulltext.")

    pubs: List[PubRecord] = []
    for row in pubs_rows:
        doc_key = str(row.get("doc_key") or "").strip()
        if not doc_key:
            continue

        if not force and doc_key in existing_keys:
            # Already have full_text for this doc
            continue

        pdf_filename = row.get("pdf_filename")
        if not pdf_filename:
            continue

        title = str(row.get("title") or "").strip()
        doc_type = row.get("doc_type")
        date = row.get("date")
        pubs.append(
            PubRecord(
                doc_key=doc_key,
                title=title,
                doc_type=str(doc_type) if doc_type is not None else None,
                date=str(date) if date is not None else None,
                pdf_filename=str(pdf_filename),
            )
        )

    if limit is not None:
        pubs = pubs[:limit]

    print(f"[*] Will extract text for {len(pubs)} publication(s).")
    return pubs


# ---------------------------------------------------------------------------
# Text stitching
# ---------------------------------------------------------------------------


def build_full_text_from_chunks(chunks: List[Chunk]) -> str:
    """
    Given a list of Chunk objects (page_number, text, bbox...), build a single
    full_text string for the whole document.

    We keep it simple:
      - sort by page_number in ascending order
      - within a page, rely on the order already produced by extract_chunks_from_pdf
      - join chunk texts per page with newlines
      - put a page separator between pages

    We are not aggressively stripping headers/footers here; that can be
    added later if needed by using bounding boxes and frequency of repeated
    lines across pages.
    """
    if not chunks:
        return ""

    # sort by page_number just in case
    chunks_sorted = sorted(chunks, key=lambda c: c.page_number)

    full_parts: List[str] = []
    current_page = None
    page_buf: List[str] = []

    def flush_page(page_no: Optional[int]):
        nonlocal page_buf
        if not page_buf:
            return
        page_text = "\n".join(t for t in page_buf if t.strip())
        if not page_text.strip():
            page_buf = []
            return
        if page_no is not None:
            full_parts.append(f"[Page {page_no}]\n{page_text}")
        else:
            full_parts.append(page_text)
        page_buf = []

    for chunk in chunks_sorted:
        page_no = getattr(chunk, "page_number", None)
        if current_page is None:
            current_page = page_no

        if page_no != current_page:
            flush_page(current_page)
            current_page = page_no

        txt = getattr(chunk, "text", "") or ""
        if txt.strip():
            page_buf.append(txt.strip())

    flush_page(current_page)

    return "\n\n".join(full_parts).strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(force: bool = False, limit: Optional[int] = None) -> None:
    client = get_supabase_client()

    pubs = load_publications_to_extract(limit=limit, force=force)
    if not pubs:
        print("[*] Nothing to extract. Exiting.")
        return

    base = Path(".")
    tmp_pdf_dir = base / "tmp_fincen_pdfs"
    tmp_pdf_dir.mkdir(exist_ok=True)

    processed = 0

    for idx, pub in enumerate(pubs, start=1):
        print(
            f"\n[*] [{idx}/{len(pubs)}] {pub.title or pub.doc_key} "
            f"({pub.doc_type}), pdf={pub.pdf_filename}"
        )

        local_pdf_path = tmp_pdf_dir / pub.pdf_filename

        # Download from Supabase bucket
        try:
            download_pdf_from_bucket(
                client,
                pdf_filename=pub.pdf_filename,
                doc_type=pub.doc_type or "",
                dest_path=local_pdf_path,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Failed to download {pub.pdf_filename}: {e}", file=sys.stderr)
            continue

        # Extract chunks using existing PyMuPDF + OCR pipeline
        try:
            chunks = extract_chunks_from_pdf(local_pdf_path)
        except Exception as e:  # noqa: BLE001
            print(
                f"  [WARN] Failed to extract chunks from {pub.pdf_filename}: {e}",
                file=sys.stderr,
            )
            continue

        if not chunks:
            print("  [WARN] No text chunks extracted; skipping.")
            continue

        full_text = build_full_text_from_chunks(chunks)
        if not full_text.strip():
            print("  [WARN] Full text is empty after stitching; skipping.")
            continue

        # Upsert into fincen_fulltext
        record = {
            "doc_key": pub.doc_key,
            "title": pub.title,
            "doc_type": pub.doc_type,
            "date": pub.date,
            "full_text": full_text,
        }

        try:
            client.table("fincen_fulltext").upsert(record).execute()
            print("  [✓] Saved full_text to fincen_fulltext.")
            processed += 1
        except Exception as e:  # noqa: BLE001
            print(
                f"  [WARN] Failed to upsert into fincen_fulltext for {pub.doc_key}: {e}",
                file=sys.stderr,
            )
            continue

        # Small delay just to be gentle on disk + Supabase
        time.sleep(0.2)

    print(f"\n[*] Done. Successfully extracted text for {processed} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract full text for FinCEN publications into fincen_fulltext."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if fincen_fulltext already has a row for this doc_key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing).",
    )

    args = parser.parse_args()
    main(force=args.force, limit=args.limit)
