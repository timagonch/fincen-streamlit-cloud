# supabase_helpers.py
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict, Any

from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Load .env (walk up directories to find it)
# ---------------------------------------------------------------------------

env_path = find_dotenv()
print(f"[supabase_helpers] Loading .env from: {env_path!r}")
load_dotenv(env_path, override=False)

# Name of your Supabase storage bucket with the PDFs
BUCKET_NAME = "fincen-pdfs"


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    """
    Build and return a Supabase client.

    Environment variables supported:

        SUPABASE_URL                 (required)
        SUPABASE_SERVICE_ROLE_KEY    (preferred, if present)
        SUPABASE_KEY                 (fallback if service-role key not set)
    """
    url: Optional[str] = os.getenv("SUPABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_URL is not set. Add it to your .env file.")

    key: Optional[str] = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    )
    if not key:
        raise RuntimeError(
            "Neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_KEY is set.\n"
            "Add at least SUPABASE_KEY to your .env file."
        )

    # Back to the standard supabase-py client creation
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Utility: title/date keys for advisories
# ---------------------------------------------------------------------------

def normalize_title_for_key(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def load_existing_advisory_keys_from_supabase(client: Client) -> Set[Tuple[str, str]]:
    """
    Returns a set of (normalized_title, date) from Supabase for doc_type='Advisory'.
    Used for early-stop: if we hit a (title, date) already present, we stop crawling.
    """
    keys: Set[Tuple[str, str]] = set()

    resp = (
        client.table("fincen_publications")
        .select("title, date")
        .eq("doc_type", "Advisory")
        .execute()
    )

    for row in resp.data or []:
        title_norm = normalize_title_for_key(row.get("title") or "")
        date_str = (row.get("date") or "").strip()
        if title_norm and date_str:
            keys.add((title_norm, date_str))

    return keys


# ---------------------------------------------------------------------------
# Existing fincen_ids for Alerts / Notices
# ---------------------------------------------------------------------------

def load_existing_ids_from_supabase(client: Client, doc_type: str) -> Set[str]:
    """
    Returns all fincen_id values for a given doc_type ('Alert' or 'Notice').
    """
    ids: Set[str] = set()

    resp = (
        client.table("fincen_publications")
        .select("fincen_id")
        .eq("doc_type", doc_type)
        .execute()
    )

    for row in resp.data or []:
        fid = (row.get("fincen_id") or "").strip()
        if fid:
            ids.add(fid)

    return ids


# ---------------------------------------------------------------------------
# Inserts / upserts into fincen_publications
# ---------------------------------------------------------------------------

def upsert_publication(
    client: Client,
    *,
    fincen_id: str,
    title: str,
    date: str,
    doc_type: str,
    pdf_filename: str,
    pdf_url: str,
    detail_url: Optional[str],
    sha256: Optional[str] = None,
) -> None:
    """
    Insert or update a single publication row in `fincen_publications`.

    We use (doc_type, pdf_filename) as the conflict target so that:
      - If a row already exists for that doc_type+pdf_filename, we update it
        (e.g., to fill in fincen_id or corrected metadata).
      - Otherwise, we insert a new row.
    """
    row = {
        "fincen_id": fincen_id or None,
        "title": title,
        "date": date,
        "doc_type": doc_type,
        "pdf_filename": pdf_filename,
        "pdf_url": pdf_url,
        "detail_url": detail_url,
        "sha256": sha256,
    }

    client.table("fincen_publications").upsert(
        row,
        on_conflict="doc_type,pdf_filename",
    ).execute()


# ---------------------------------------------------------------------------
# Storage: upload PDFs to bucket
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def upload_pdf_to_bucket(
    client: Client,
    local_path: Path,
    remote_path: str,
) -> str:
    """
    Upload a local PDF to Supabase Storage (BUCKET_NAME) with upsert=true.

    remote_path example:
        "advisories/FIN-2024-ADVISORY1.pdf"
    """
    storage = client.storage
    with open(local_path, "rb") as f:
        storage.from_(BUCKET_NAME).upload(
            remote_path,
            f,
            {"content-type": "application/pdf", "upsert": "true"},
        )
    return remote_path


# ---------------------------------------------------------------------------
# Mapper helpers: fetch publications + download PDFs + replace tables
# ---------------------------------------------------------------------------

def fetch_publications_for_mapper(client: Client) -> List[Dict[str, Any]]:
    """
    Return publications that have not yet been processed by the mapper
    (mapper_processed_at IS NULL).
    """
    resp = (
        client.table("fincen_publications")
        .select("fincen_id, title, date, doc_type, pdf_filename")
        .is_("mapper_processed_at", None)
        .execute()
    )
    return resp.data or []


def download_pdf_from_bucket(
    client: Client,
    pdf_filename: str,
    doc_type: str,
    dest_path: Path,
) -> None:
    """
    Download a PDF from Supabase Storage into dest_path.
    Assumes bucket layout:
        advisories/<pdf_filename>
        alerts/<pdf_filename>
        notices/<pdf_filename>
    """
    folder = {
        "Advisory": "advisories",
        "Alert": "alerts",
        "Notice": "notices",
    }.get(doc_type, "other")

    remote_path = f"{folder}/{pdf_filename}"

    data = client.storage.from_(BUCKET_NAME).download(remote_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)


def replace_table_rows(client: Client, table_name: str, rows: List[Dict[str, Any]]) -> None:
    """
    Legacy full-refresh helper for mapper outputs (not used by the
    incremental OCR mapper). Kept here in case you want to do a
    full table rebuild from local CSVs.
    """
    # Clear table
    client.table(table_name).delete().neq("article_name", "").execute()

    # Chunk inserts to avoid payload limits
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        if batch:
            client.table(table_name).insert(batch).execute()
