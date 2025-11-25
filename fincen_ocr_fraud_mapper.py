#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_ocr_fraud_mapper.py (Supabase + Incremental Semantic Mapper + OCR Fallback)
----------------------------------------------------------------------------------

Pipeline:

INPUT:
  Supabase:
    - Table: fincen_publications
        * fincen_id
        * title
        * date
        * doc_type        ('Advisory' | 'Alert' | 'Notice')
        * pdf_filename
        * mapper_processed_at (timestamptz, NULL = not processed yet)

    - Storage bucket: fincen-pdfs
        * advisories/<pdf_filename>
        * alerts/<pdf_filename>
        * notices/<pdf_filename>

OUTPUT:
  Supabase tables:
    - fincen_fraud_mapping      (document-level semantic fraud summary)
    - fincen_semantic_chunks    (chunk-level semantic tagging + bounding boxes)
    - fincen_keyword_locations  (per-fraud-type highlight rectangles)

Behavior:
  - Only processes publications where mapper_processed_at IS NULL.
  - For each doc:
      * Download PDF from bucket to a temp folder
      * Extract text chunks + normalized bounding boxes via PyMuPDF
          - If a page has no extractable text → fallback to PaddleOCR
      * Tag chunks with fraud families via SentenceTransformers + cosine
      * Write/overwrite rows for that doc in all three mapper tables
      * Set mapper_processed_at = now() in fincen_publications

Run locally:
    uv run fincen_ocr_fraud_mapper.py
"""

from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from supabase_helpers import get_supabase_client, BUCKET_NAME

# Optional OCR (PaddleOCR)
try:
    from paddleocr import PaddleOCR

    _OCR_AVAILABLE = True
    OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en")
except Exception:
    _OCR_AVAILABLE = False
    OCR_ENGINE = None


# ---------------------------------------------------------------------------
# FRAUD FAMILY DEFINITIONS (semantic labels)
# ---------------------------------------------------------------------------

FRAUD_FAMILIES: Dict[str, str] = {
    "sanctions_evasion": "Sanctions evasion, including use of shell companies, third-country intermediaries, or complex structures to move funds around sanctions.",
    "terrorist_financing": "Terrorist financing, including fundraising, funnel accounts, charities misused for terrorism, and foreign fighter support.",
    "human_trafficking": "Human trafficking or human exploitation, including forced labor, sexual exploitation, or related financial activity.",
    "human_smuggling": "Human smuggling across borders, with payments to smugglers or facilitators moving people illegally.",
    "romance_elder_scam": "Romance scams or elder fraud schemes targeting older or vulnerable persons for financial gain.",
    "synthetic_identity": "Synthetic identity schemes that combine real and fake identity elements to open accounts or obtain credit.",
    "virtual_assets_crypto": "Use of virtual assets or cryptocurrency such as Bitcoin, stablecoins, mixers, or DeFi protocols to conduct illicit finance.",
    "trade_based_money_laundering": "Trade-based money laundering through invoices, over/under-invoicing, phantom shipments, or misuse of import-export flows.",
    "corruption_bribery": "Corruption, bribery, embezzlement, or misuse of public office for private gain.",
    "ransomware_cyber": "Ransomware, cyber extortion, malware, or related cyber-enabled financial crime.",
    "fraud_government_programs": "Fraud against government benefit or relief programs such as unemployment insurance, PPP, EIP, healthcare benefits.",
    "bulk_cash_smuggling": "Bulk cash smuggling or cross-border movement of large amounts of currency or monetary instruments.",
    "shell_front_companies": "Use of shell companies, front companies, or complex corporate structures to hide ownership or illicit proceeds.",
    "correspondent_banking": "Misuse of correspondent banking relationships, nested accounts, or high-risk foreign financial institutions.",
    "money_mules": "Use of money mules or intermediary accounts to move or layer illicit funds.",
    "card_fraud": "Payment card fraud, including debit or credit card compromise, card-not-present fraud, or related schemes.",
}


# ---------------------------------------------------------------------------
# EMBEDDINGS + COSINE SIMILARITY
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    # Hash for embedding cache key
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _cosine(v1, v2) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    dot = float((a * b).sum())
    n1 = float(np.linalg.norm(a))
    n2 = float(np.linalg.norm(b))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def get_embed_fn_with_cache(model_name: str, cache_path: Path):
    """
    Returns an embed(texts) -> List[List[float]] function backed by a JSON cache.

    - Uses SentenceTransformer(model_name).
    - Caches embeddings by MD5 hash of text.
    - Cache file is kept in .cache/embed_cache.json by default (git-ignore-friendly).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)

    if cache_path.exists():
        try:
            cache: Dict[str, List[float]] = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    else:
        cache = {}

    def embed(texts: List[str]) -> List[List[float]]:
        nonlocal cache
        results: List[List[float]] = []
        to_compute: List[str] = []
        idx_map: List[int] = []

        for i, t in enumerate(texts):
            h = _hash_text(t)
            if h in cache:
                results.append(cache[h])
            else:
                results.append([])
                to_compute.append(t)
                idx_map.append(i)

        if to_compute:
            new_vecs = model.encode(to_compute, show_progress_bar=False).tolist()
            for i, vec in zip(idx_map, new_vecs):
                h = _hash_text(texts[i])
                cache[h] = vec
                results[i] = vec
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

        return results

    return embed


def build_fraud_family_vectors(embed_fn) -> Dict[str, List[float]]:
    """
    Compute an embedding vector for each fraud family description.
    Returns: {fraud_key: embedding_vector}
    """
    fraud_vecs: Dict[str, List[float]] = {}
    texts = list(FRAUD_FAMILIES.values())
    keys = list(FRAUD_FAMILIES.keys())
    vecs = embed_fn(texts)
    for k, v in zip(keys, vecs):
        fraud_vecs[k] = v
    return fraud_vecs


def tag_chunk_with_fraud_types(
    text: str,
    fraud_vecs: Dict[str, List[float]],
    embed_fn,
    sim_threshold: float = 0.42,
    top_k: int = 3,
):
    """
    For a given chunk of text:
      - Embed the chunk
      - Compute cosine similarity to each fraud family
      - Return:
          * fraud_types: list of fraud keys with similarity >= threshold (up to top_k)
          * scores: {fraud_key: similarity}
    """
    text = text.strip()
    if not text:
        return [], {}

    vec = embed_fn([text])[0]
    scores: Dict[str, float] = {}
    for k, fraud_vec in fraud_vecs.items():
        scores[k] = _cosine(vec, fraud_vec)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: List[str] = []
    for k, v in sorted_items[:top_k]:
        if v >= sim_threshold:
            selected.append(k)

    # Deduplicate while preserving order
    selected = list(dict.fromkeys(selected))
    return selected, scores


# ---------------------------------------------------------------------------
# OCR FALLBACK
# ---------------------------------------------------------------------------

def ocr_page_to_text(page) -> str:
    """
    Render a PyMuPDF page to PNG bytes and run PaddleOCR to get text.

    Returns a concatenated string of recognized lines, or "" if OCR is
    unavailable or fails.
    """
    if not _OCR_AVAILABLE or OCR_ENGINE is None:
        return ""

    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")

    try:
        results = OCR_ENGINE.ocr(img_bytes, cls=True)
        lines: List[str] = []
        for res in results:
            for line in res:
                # line[1][0] is the recognized text string
                lines.append(line[1][0])
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# CHUNKING WITH PyMuPDF + OCR FALLBACK
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    article_name: str
    page_number: int
    bbox: Tuple[float, float, float, float]   # normalized (x0, y0, x1, y1)
    text: str


@dataclass
class SemanticChunk:
    article_name: str
    fincen_id: Optional[str]
    doc_type: Optional[str]
    page_number: int
    chunk_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    is_high_signal: bool
    fraud_types_semantic: List[str]
    fraud_scores_semantic: Dict[str, float]


@dataclass
class DocSummary:
    article_name: str
    fincen_id: Optional[str]
    title: Optional[str]
    date: Optional[str]
    doc_type: Optional[str]
    primary_fraud_type_semantic: str
    secondary_fraud_types_semantic: List[str]
    fraud_mention_counts_semantic: Dict[str, int]
    total_chunks: int


def extract_chunks_from_pdf(pdf_path: Path, max_chunk_chars: int = 900) -> List[Chunk]:
    """
    Use PyMuPDF to:
      - Iterate pages
      - Get text blocks
      - Merge into paragraph-level chunks up to max_chunk_chars
      - Normalize bounding boxes to [0, 1] x [0, 1]
      - If a page has no text blocks, fallback to OCR (PaddleOCR) and
        chunk OCR text with a full-page bounding box.
    """
    chunks: List[Chunk] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_rect = page.rect
            W, H = float(page_rect.width), float(page_rect.height)

            blocks = page.get_text("blocks") or []
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            page_has_text = False

            para_buf: List[str] = []
            x0 = y0 = math.inf
            x1 = y1 = -math.inf
            current_len = 0

            def flush_buf():
                nonlocal para_buf, x0, y0, x1, y1, current_len
                txt = " ".join(para_buf).strip()
                if not txt:
                    para_buf = []
                    current_len = 0
                    return

                if W <= 0 or H <= 0 or not math.isfinite(W) or not math.isfinite(H):
                    bbox_norm = (0.0, 0.0, 1.0, 1.0)
                else:
                    bbox_norm = (x0 / W, y0 / H, x1 / W, y1 / H)

                chunks.append(
                    Chunk(
                        article_name=pdf_path.name,
                        page_number=page_index + 1,
                        bbox=bbox_norm,
                        text=txt,
                    )
                )
                para_buf = []
                current_len = 0

            # ----- Normal text-block extraction -----
            for b in blocks:
                bx0, by0, bx1, by1, bt, *_ = b
                txt = (bt or "").strip()
                if not txt:
                    continue

                page_has_text = True

                if current_len + len(txt) > max_chunk_chars and para_buf:
                    flush_buf()
                    x0, y0, x1, y1 = bx0, by0, bx1, by1

                para_buf.append(txt)
                current_len += len(txt)

                x0 = min(x0, bx0)
                y0 = min(y0, by0)
                x1 = max(x1, bx1)
                y1 = max(y1, by1)

            flush_buf()

            # ----- OCR fallback if no text on page -----
            if not page_has_text and _OCR_AVAILABLE:
                ocr_text = ocr_page_to_text(page)
                if ocr_text.strip():
                    paragraphs = [p.strip() for p in ocr_text.split("\n") if p.strip()]
                    buf: List[str] = []
                    length = 0

                    for p in paragraphs:
                        if length + len(p) > max_chunk_chars and buf:
                            chunks.append(
                                Chunk(
                                    article_name=pdf_path.name,
                                    page_number=page_index + 1,
                                    bbox=(0.0, 0.0, 1.0, 1.0),
                                    text=" ".join(buf).strip(),
                                )
                            )
                            buf = []
                            length = 0

                        buf.append(p)
                        length += len(p) + 1

                    if buf:
                        chunks.append(
                            Chunk(
                                article_name=pdf_path.name,
                                page_number=page_index + 1,
                                bbox=(0.0, 0.0, 1.0, 1.0),
                                text=" ".join(buf).strip(),
                            )
                        )

    return chunks


# ---------------------------------------------------------------------------
# AGGREGATE TO DOC-LEVEL + TABLE ROWS
# ---------------------------------------------------------------------------

def aggregate_document(
    article_name: str,
    fincen_id: Optional[str],
    title: Optional[str],
    date: Optional[str],
    doc_type: Optional[str],
    chunks: List[Chunk],
    fraud_results: List[Tuple[List[str], Dict[str, float]]],
):
    """
    From chunk-level fraud tagging, compute:

      - DocSummary (for fincen_fraud_mapping)
      - List[SemanticChunk] (for fincen_semantic_chunks)
      - keyword_rows (for fincen_keyword_locations)
    """
    mention_counts: Counter = Counter()
    score_sums: Counter = Counter()
    semantic_chunks: List[SemanticChunk] = []
    keyword_rows: List[Dict[str, Any]] = []

    for idx, (ch, (f_types, f_scores)) in enumerate(zip(chunks, fraud_results)):
        if not f_types:
            is_high = False
        else:
            max_score = max(f_scores.values()) if f_scores else 0.0
            is_high = max_score >= 0.5

        for ft in f_types:
            mention_counts[ft] += 1
            score_sums[ft] += float(f_scores.get(ft, 0.0))

        sc = SemanticChunk(
            article_name=article_name,
            fincen_id=fincen_id,
            doc_type=doc_type,
            page_number=ch.page_number,
            chunk_index=idx,
            bbox=ch.bbox,
            text=ch.text,
            is_high_signal=is_high,
            fraud_types_semantic=f_types,
            fraud_scores_semantic=f_scores,
        )
        semantic_chunks.append(sc)

        # High-signal chunks contribute highlight rectangles
        if is_high and f_types:
            for ft in f_types:
                keyword_rows.append(
                    {
                        "article_name": article_name,
                        "fincen_id": fincen_id,
                        "doc_type": doc_type,
                        "fraud_type": ft,
                        "page_number": ch.page_number,
                        "x0_norm": ch.bbox[0],
                        "y0_norm": ch.bbox[1],
                        "x1_norm": ch.bbox[2],
                        "y1_norm": ch.bbox[3],
                    }
                )

    if mention_counts:
        sorted_scores = sorted(score_sums.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_scores[0][0]
        primary_score = sorted_scores[0][1]
        secondary = [k for (k, v) in sorted_scores[1:] if v >= 0.4 * primary_score]
    else:
        primary = ""
        secondary = []

    summary = DocSummary(
        article_name=article_name,
        fincen_id=fincen_id,
        title=title,
        date=date,
        doc_type=doc_type,
        primary_fraud_type_semantic=primary,
        secondary_fraud_types_semantic=secondary,
        fraud_mention_counts_semantic=dict(mention_counts),
        total_chunks=len(chunks),
    )

    return summary, semantic_chunks, keyword_rows


# ---------------------------------------------------------------------------
# SUPABASE HELPERS (specific to mapper)
# ---------------------------------------------------------------------------

def fetch_unprocessed_publications(client):
    """
    Return publications that have not yet been processed by the mapper
    (mapper_processed_at IS NULL).

    After the one-time full rebuild, this keeps the mapper incremental:
    only new rows in fincen_publications (or ones you manually reset)
    will be processed.
    """
    resp = (
        client.table("fincen_publications")
        .select("fincen_id, title, date, doc_type, pdf_filename")
        .is_("mapper_processed_at", None)
        .execute()
    )
    return resp.data or []




def download_pdf_from_bucket(client, pdf_filename: str, doc_type: str, dest_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main() -> None:
    client = get_supabase_client()

    print("[*] Fetching unprocessed publications from Supabase...")
    pubs = fetch_unprocessed_publications(client)
    if not pubs:
        print("[✓] No unprocessed publications. Mapper is up-to-date.")
        return

    print(f"[*] {len(pubs)} publication(s) to process.")

    base = Path(".")
    tmp_pdf_dir = base / "tmp_fincen_pdfs"
    tmp_pdf_dir.mkdir(exist_ok=True)

    # Embedding cache in a git-ignore-friendly folder
    cache_path = base / ".cache" / "embed_cache.json"
    embed_fn = get_embed_fn_with_cache("all-MiniLM-L6-v2", cache_path)
    fraud_vecs = build_fraud_family_vectors(embed_fn)

    for idx, pub in enumerate(pubs, start=1):
        pdf_filename = (pub.get("pdf_filename") or "").strip()
        if not pdf_filename:
            print(f"[{idx}/{len(pubs)}] Skipping: missing pdf_filename")
            continue

        article_name = pub.get("pdf_filename").strip()
        fincen_id = pub.get("fincen_id")
        title = pub.get("title")
        date = pub.get("date")
        doc_type = pub.get("doc_type")

        print(f"\n[{idx}/{len(pubs)}] {article_name} ({doc_type})")

        local_pdf_path = tmp_pdf_dir / pdf_filename
        try:
            download_pdf_from_bucket(client, pdf_filename, doc_type, local_pdf_path)
        except Exception as e:
            print(f"  [WARN] Failed to download {pdf_filename} from bucket: {e}")
            continue

        try:
            chunks = extract_chunks_from_pdf(local_pdf_path)
        except Exception as e:
            print(f"  [WARN] Failed to extract chunks from {pdf_filename}: {e}")
            continue

        print(f"  -> {len(chunks)} chunks")

        if not chunks:
            print("  [WARN] No chunks extracted (even with OCR). Skipping.")
            continue

        fraud_results: List[Tuple[List[str], Dict[str, float]]] = []
        for ch in chunks:
            f_types, f_scores = tag_chunk_with_fraud_types(ch.text, fraud_vecs, embed_fn)
            fraud_results.append((f_types, f_scores))

        summary, sem_chunks, kw_rows = aggregate_document(
            article_name, fincen_id, title, date, doc_type, chunks, fraud_results
        )

        # -------------------------
        # Write to Supabase tables
        # -------------------------

        # Delete existing rows for this article (idempotent per-doc refresh)
        client.table("fincen_fraud_mapping").delete().eq("article_name", article_name).execute()
        client.table("fincen_semantic_chunks").delete().eq("article_name", article_name).execute()
        client.table("fincen_keyword_locations").delete().eq("article_name", article_name).execute()

        # Document-level row
        doc_row = {
            "article_name": summary.article_name,
            "fincen_id": summary.fincen_id,
            "title": summary.title,
            "date": summary.date,
            "doc_type": summary.doc_type,
            "primary_fraud_type_semantic": summary.primary_fraud_type_semantic,
            "secondary_fraud_types_semantic": summary.secondary_fraud_types_semantic,
            "fraud_mention_counts_semantic": summary.fraud_mention_counts_semantic,
            "total_chunks": summary.total_chunks,
        }
        client.table("fincen_fraud_mapping").insert(doc_row).execute()

        # Chunk-level rows
        if sem_chunks:
            chunk_rows: List[Dict[str, Any]] = []
            for sc in sem_chunks:
                chunk_rows.append(
                    {
                        "article_name": sc.article_name,
                        "fincen_id": sc.fincen_id,
                        "doc_type": sc.doc_type,
                        "page_number": sc.page_number,
                        "chunk_index": sc.chunk_index,
                        "x0_norm": sc.bbox[0],
                        "y0_norm": sc.bbox[1],
                        "x1_norm": sc.bbox[2],
                        "y1_norm": sc.bbox[3],
                        "text": sc.text,
                        "is_high_signal": sc.is_high_signal,
                        "fraud_types_semantic": sc.fraud_types_semantic,
                        "fraud_scores_semantic": sc.fraud_scores_semantic,
                    }
                )
            client.table("fincen_semantic_chunks").insert(chunk_rows).execute()

        # Keyword-locations rows
        if kw_rows:
            client.table("fincen_keyword_locations").insert(kw_rows).execute()

        # Mark this publication as processed
        now_ts = datetime.now(timezone.utc).isoformat()
        client.table("fincen_publications").update(
            {"mapper_processed_at": now_ts}
        ).eq("pdf_filename", pdf_filename).eq("doc_type", doc_type).execute()

        print("  [✓] Mapper rows written and publication marked as processed.")

    print("\n[✓] Mapper complete. All new publications processed.")


if __name__ == "__main__":
    main()
