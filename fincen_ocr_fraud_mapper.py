#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_ocr_fraud_mapper.py (Semantic-First Version)
---------------------------------------------------

Run (example):

    uv run python fincen_ocr_fraud_mapper.py `
        fincen_advisory_pdfs `
        fincen_fraud_mapping.csv `
        fincen_fraud_mapping_details.json `
        fincen_keyword_locations.csv `
        fincen_semantic_chunks.csv `
        --extra-pdf-dir fincen_alert_pdfs `
        --extra-pdf-dir fincen_notice_pdfs

High-level purpose
------------------
This script builds the **semantic backbone** for the FinCEN Fraud Intelligence
Platform. It does the heavy lifting over FinCEN PDFs (advisories, alerts,
notices) *once*, and persists the results in CSV/JSON files that are reused
by downstream steps:

- LLM summarization (`fincen_summary_generator.py`)
- Streamlit analytics / dashboarding
- RAG / semantic search

What this script does
---------------------
For each PDF in your advisory/alert/notice folders, it:

1. Extracts text blocks with page-level geometry using PyMuPDF.
   - Each block becomes a "chunk" with:
       - article_name (PDF filename)
       - page_number
       - bbox (normalized [0,1] coords: x0, y0, x1, y1)
       - text

2. Splits overly long blocks into smaller chunks (to keep embedding inputs
   manageable, ~1–3 paragraphs).

3. Uses a local sentence-embedding model to semantically compare each
   chunk to a fixed set of fraud "families" (money laundering, TBML, etc.).
   - For each chunk we store:
       - fraud_types_semantic: list of fraud families that match
       - fraud_scores_semantic: {fraud_type: similarity}

4. Classifies each chunk into a simple semantic "role" using keyword rules:
      - typology_description
      - red_flag_indicator
      - sar_guidance
      - regulatory_requirement
      - actor_description
      - geography_reference
      - channel_reference
      - sector_reference
      - case_example
      - context
   These roles are inferred using simple keyword heuristics (no LLM here),
   but the schema is designed so you can later upgrade to LLM-based
   classification without changing the outputs.

5. Extracts cheap entity-like hints per chunk using keyword lists:
      - countries_mentioned
      - sectors_mentioned
      - channels_mentioned
      - programs_mentioned
      - actors_mentioned

6. Aggregates per-document fraud signals and chunk counts to determine:
      - primary_fraud_type_semantic (top-1 by intensity)
      - secondary_fraud_types_semantic (top 2–3)
      - semantic_mention_counts_json (JSON map: {fraud_type: count})
      - total_chunks
      - high_signal_chunks

7. Produces four outputs:

(1) fincen_fraud_mapping.csv  (DOCUMENT-LEVEL)
    - One row per PDF with:
        article_name (PDF filename)
        primary_fraud_type_semantic
        secondary_fraud_types_semantic
        semantic_mention_counts_json
        total_chunks
        high_signal_chunks
      plus any existing columns (doc_title, doc_date, doc_type, etc.)
      if the file already existed.

(2) fincen_fraud_mapping_details.json  (DOCUMENT-LEVEL, DETAILED)
    - Keyed by article_name.
    - For each doc:
        {
          "article_name": ...,
          "fraud_type_scores": {fraud_type: aggregate_score},
          "semantic_mention_counts": {fraud_type: count_of_chunks},
          "high_signal_chunk_ids": [[page_number, chunk_index], ...]
        }
      - This is useful for debugging, offline analysis, and future RAG uses.

(3) fincen_keyword_locations.csv  (SEMANTIC HIGHLIGHT LOCATIONS)
   - Despite the name, this is now **semantic**, not regex-based.
   - One row per high-signal chunk x fraud-type pair.
   - Contains article_name, fincen_id, fraud_type, page_number, and bbox.
   - Can be used by the UI to highlight the most relevant paragraphs on
     each PDF page for a given fraud type.

(4) fincen_semantic_chunks.csv  (CHUNK-LEVEL)
   - One row per chunk of text on each page of each PDF.
   - Fields include:
       article_name,
       fincen_id (if available),
       page_number,
       chunk_index,
       bbox (x0_norm, y0_norm, x1_norm, y1_norm),
       text,
       chunk_role,
       is_high_signal,
       fraud_types_semantic,
       fraud_scores_semantic,
       countries_mentioned,
       sectors_mentioned,
       channels_mentioned,
       programs_mentioned,
       actors_mentioned.

OCR behavior
------------
- For most modern FinCEN 508 PDFs, `get_text("blocks")` returns text.
- If a page has **no extractable text** and PaddleOCR is installed, the
  script:
    - Renders the page as an image.
    - Runs OCR.
    - Treats the entire page as a full-page chunk (bbox = (0,0,1,1)),
      potentially split into multiple chunks if very long.

- If PaddleOCR is not installed, OCR fallback is disabled and image-only
  pages will remain unprocessed (same behavior as earlier versions).

"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import tempfile
from uuid import uuid4

import fitz  # PyMuPDF
import pandas as pd
import argparse 

try:
    from paddleocr import PaddleOCR  # type: ignore
    _OCR_AVAILABLE = True
    OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en")
except Exception:
    _OCR_AVAILABLE = False
    OCR_ENGINE = None

# ------------------------------ CONFIG ---------------------------------------

# Default paths (can be overridden via CLI)
OUT_DOC_CSV = "fincen_fraud_mapping.csv"
OUT_DETAILS_JSON = "fincen_fraud_mapping_details.json"
OUT_KEYWORD_LOCATIONS = "fincen_keyword_locations.csv"
OUT_SEMANTIC_CHUNKS = "fincen_semantic_chunks.csv"

# Advisory/Alert/Notice metadata CSVs – if available, we merge titles/dates
DEFAULT_META_CSVS = [
    "fincen_advisories.csv",
    "fincen_alerts.csv",
    "fincen_notices.csv",
]

# ----------------------------- FRAUD ONTOLOGY --------------------------------

# Canonical fraud types for semantic tagging
FRAUD_TYPES: Dict[str, str] = {
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

# --------------------------- SIMPLE ENTITY LEXICONS --------------------------

COUNTRY_KEYWORDS = {
    "united states": "US",
    "u.s.": "US",
    "u.s": "US",
    "usa": "US",
    "russia": "RU",
    "ukraine": "UA",
    "iran": "IR",
    "north korea": "KP",
    "china": "CN",
    "hong kong": "HK",
    "mexico": "MX",
    "canada": "CA",
    "venezuela": "VE",
    "colombia": "CO",
    "panama": "PA",
    "cayman islands": "KY",
    "british virgin islands": "VG",
    "united kingdom": "GB",
    "u.k.": "GB",
    "uk": "GB",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "turkey": "TR",
    "uae": "AE",
    "united arab emirates": "AE",
    "saudi arabia": "SA",
}

SECTOR_KEYWORDS = {
    "bank": "bank",
    "banks": "bank",
    "credit union": "credit_union",
    "credit unions": "credit_union",
    "money services business": "msb",
    "msb": "msb",
    "broker-dealer": "broker_dealer",
    "casino": "casino",
    "card club": "card_club",
    "insurance": "insurance",
    "fintech": "fintech",
}

CHANNEL_KEYWORDS = {
    "wire transfer": "wire",
    "wires": "wire",
    "wire transfers": "wire",
    "ach": "ach",
    "ach transfer": "ach",
    "cash deposit": "cash_deposit",
    "cash deposits": "cash_deposit",
    "atm": "atm",
    "atm withdrawal": "atm",
    "check": "check",
    "checks": "check",
    "remote deposit": "remote_deposit",
    "mobile deposit": "remote_deposit",
    "online banking": "online_banking",
    "prepaid card": "prepaid_card",
    "prepaid cards": "prepaid_card",
    "card-to-card": "card_to_card",
}

PROGRAM_KEYWORDS = {
    "ppp loan": "ppp",
    "paycheck protection program": "ppp",
    "economic impact payment": "eip",
    "eip": "eip",
    "unemployment insurance": "ui",
    "unemployment benefits": "ui",
    "snap": "snap",
    "medicare": "medicare",
    "medicaid": "medicaid",
}

ACTOR_KEYWORDS = {
    "money mule": "money_mule",
    "mules": "money_mule",
    "shell company": "shell_company",
    "front company": "front_company",
    "front business": "front_company",
    "armored car service": "acs",
    "armoured car service": "acs",
    "cash courier": "courier",
    "courier": "courier",
}

# --------------------------------- TYPES -------------------------------------


@dataclass
class Chunk:
    article_name: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    text: str


@dataclass
class SemanticChunk:
    article_name: str
    page_number: int
    chunk_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    chunk_role: str
    is_high_signal: bool
    fraud_types_semantic: List[str]
    fraud_scores_semantic: Dict[str, float]
    countries_mentioned: List[str]
    sectors_mentioned: List[str]
    channels_mentioned: List[str]
    programs_mentioned: List[str]
    actors_mentioned: List[str]


@dataclass
class DocSemanticSummary:
    article_name: str
    primary_fraud_type_semantic: str
    secondary_fraud_types_semantic: str  # pipe-separated
    semantic_mention_counts_json: str    # JSON string of {fraud_type: count}
    total_chunks: int
    high_signal_chunks: int


@dataclass
class DocDetailedStats:
    article_name: str
    fraud_type_scores: Dict[str, float]
    semantic_mention_counts: Dict[str, int]
    high_signal_chunk_ids: List[Tuple[int, int]]  # (page_number, chunk_index)


# -------------------------- EMBEDDING + CACHE --------------------------------


def _hash_text(text: str) -> str:
    return md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _get_cached_embed_fn(model_name: str, cache_path: Path):
    """
    Returns an embedding function that uses SentenceTransformers + a simple
    JSON file cache keyed by MD5 of the text. This avoids recomputing
    embeddings for repeated chunks across runs.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    if cache_path.exists():
        try:
            cache: Dict[str, List[float]] = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    else:
        cache = {}

    def embed(texts: List[str]) -> List[List[float]]:
        nonlocal cache
        results: List[List[float]] = []
        to_compute: List[str] = []
        missing_idx: List[int] = []

        for i, t in enumerate(texts):
            h = _hash_text(t)
            if h in cache:
                results.append(cache[h])
            else:
                results.append([])  # placeholder
                to_compute.append(t)
                missing_idx.append(i)

        if to_compute:
            vecs = model.encode(to_compute, normalize_embeddings=True)
            for idx, vec in zip(missing_idx, vecs.tolist()):
                h = _hash_text(texts[idx])
                results[idx] = vec
                cache[h] = vec

            try:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except Exception:
                # best-effort cache write; do not crash the pipeline
                pass

        return results

    return embed


# ---------------------------- PDF CHUNK EXTRACTION ---------------------------

MAX_CHUNK_CHARS = 1400


def extract_chunks_from_pdf(pdf_path: Path) -> List[Chunk]:
    """
    Extracts text chunks from a PDF using PyMuPDF 'blocks' as a starting point,
    with an OCR fallback for image-only pages.

    - Each text block on each page becomes a Chunk, with normalized bbox and text.
    - We perform minimal cleaning and enforce a maximum character length per
      chunk (splitting large blocks into smaller paragraphs if needed).
    - If a page has no extractable text and PaddleOCR is available, we render
      the page to an image and run OCR, treating the entire page as one or more
      chunks with a full-page bbox (0,0,1,1).
    """
    doc = fitz.open(pdf_path)
    chunks: List[Chunk] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        W, H = float(page.rect.width), float(page.rect.height)

        page_has_text = False

        for blk in blocks:
            # blk: (x0, y0, x1, y1, text, block_no, block_type, ...)
            if len(blk) < 5:
                continue
            x0, y0, x1, y1, txt = float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]), blk[4]
            txt = (txt or "").strip()
            if not txt:
                continue

            page_has_text = True

            # Split very long blocks into smaller chunks by paragraph
            paragraphs = [p.strip() for p in txt.split("\n") if p.strip()]
            para_buf: List[str] = []
            current_len = 0
            for p in paragraphs:
                if current_len + len(p) > MAX_CHUNK_CHARS and para_buf:
                    chunk_text = " ".join(para_buf).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                article_name=pdf_path.name,
                                page_number=page_index + 1,
                                bbox=(x0 / W, y0 / H, x1 / W, y1 / H),
                                text=chunk_text,
                            )
                        )
                    para_buf = []
                    current_len = 0
                para_buf.append(p)
                current_len += len(p) + 1

            if para_buf:
                chunk_text = " ".join(para_buf).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            article_name=pdf_path.name,
                            page_number=page_index + 1,
                            bbox=(x0 / W, y0 / H, x1 / W, y1 / H),
                            text=chunk_text,
                        )
                    )

        # OCR fallback if the page had no text
        if (not page_has_text) and _OCR_AVAILABLE and OCR_ENGINE is not None:
            try:
                ocr_text = _ocr_page_to_text(page)
            except Exception:
                ocr_text = ""
            ocr_text = (ocr_text or "").strip()
            if ocr_text:
                paragraphs = [p.strip() for p in ocr_text.split("\n") if p.strip()]
                para_buf: List[str] = []
                current_len = 0
                for p in paragraphs:
                    if current_len + len(p) > MAX_CHUNK_CHARS and para_buf:
                        chunk_text = " ".join(para_buf).strip()
                        if chunk_text:
                            chunks.append(
                                Chunk(
                                    article_name=pdf_path.name,
                                    page_number=page_index + 1,
                                    bbox=(0.0, 0.0, 1.0, 1.0),
                                    text=chunk_text,
                                )
                            )
                        para_buf = []
                        current_len = 0
                    para_buf.append(p)
                    current_len += len(p) + 1

                if para_buf:
                    chunk_text = " ".join(para_buf).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                article_name=pdf_path.name,
                                page_number=page_index + 1,
                                bbox=(0.0, 0.0, 1.0, 1.0),
                                text=chunk_text,
                            )
                        )

    doc.close()
    return chunks


def _ocr_page_to_text(page) -> str:
    """
    Render a PyMuPDF page to an image and run PaddleOCR on it (if available).
    Returns concatenated text lines or an empty string if OCR is unavailable.
    """
    if not _OCR_AVAILABLE or OCR_ENGINE is None:
        return ""

    # Render the page to a temporary PNG
    pix = page.get_pixmap(dpi=200)
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"fincen_ocr_{uuid4().hex}.png"
    pix.save(tmp_path.as_posix())

    try:
        result = OCR_ENGINE.ocr(str(tmp_path), cls=True)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    lines: List[str] = []
    if not result:
        return ""
    # PaddleOCR result is a list of [ [ (box, (text, score)), ... ], ... ]
    for line in result:
        for item in line:
            if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                text = str(item[1][0]).strip()
                if text:
                    lines.append(text)

    return "\n".join(lines)


# ----------------------- CHUNK ROLE & ENTITY TAGGING -------------------------


def classify_chunk_role(text: str) -> str:
    """
    Cheap keyword-based classifier for chunk roles.
    """
    t = text.lower()

    if "red flag" in t or "red-flag" in t or "indicators of" in t:
        return "red_flag_indicator"

    if "sar" in t and ("filing" in t or "file" in t or "report suspicious" in t):
        return "sar_guidance"

    if any(kw in t for kw in ["must file", "are required to", "obligation", "must report"]):
        return "regulatory_requirement"

    if any(kw in t for kw in ["typology", "scheme", "pattern", "modus operandi"]):
        return "typology_description"

    if any(kw in t for kw in ["money mule", "mules", "perpetrator", "bad actors", "criminal networks"]):
        return "actor_description"

    if any(kw in t for kw in ["jurisdiction", "country", "region", "cross-border", "international"]):
        return "geography_reference"

    if any(kw in t for kw in ["wire transfer", "wire transfers", "cash deposits", "ach ", "prepaid card", "mobile app"]):
        return "channel_reference"

    if any(kw in t for kw in ["bank", "credit union", "money services business", "msb", "casino"]):
        return "sector_reference"

    if any(kw in t for kw in ["for example", "in one case", "in another case", "in one instance"]):
        return "case_example"

    return "context"


def extract_entities_for_chunk(text: str) -> Dict[str, List[str]]:
    """
    Very lightweight entity-like tagging from keyword lexicons.
    """
    t = text.lower()

    def _extract_from_map(mapping: Dict[str, str]) -> List[str]:
        found = set()
        for kw, label in mapping.items():
            if kw in t:
                found.add(label)
        return sorted(found)

    return {
        "countries": _extract_from_map(COUNTRY_KEYWORDS),
        "sectors": _extract_from_map(SECTOR_KEYWORDS),
        "channels": _extract_from_map(CHANNEL_KEYWORDS),
        "programs": _extract_from_map(PROGRAM_KEYWORDS),
        "actors": _extract_from_map(ACTOR_KEYWORDS),
    }


# ----------------------- FRAUD SIMILARITY PER CHUNK --------------------------


def build_fraud_type_vectors(embed_fn) -> Dict[str, List[float]]:
    """
    Embed the descriptions of each fraud type once.
    """
    keys = list(FRAUD_TYPES.keys())
    descs = [FRAUD_TYPES[k] for k in keys]
    vecs = embed_fn(descs)
    return {k: v for k, v in zip(keys, vecs)}


def tag_chunk_with_fraud_types(
    text: str,
    fraud_vecs: Dict[str, List[float]],
    embed_fn,
    sim_threshold: float = 0.42,
    top_k: int = 3,
    min_sim_floor: float = 0.30,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Compute semantic similarity between chunk text and each fraud family.
    """
    if not text.strip():
        return [], {}

    vec = embed_fn([text])[0]
    scores: Dict[str, float] = {}
    for key, fvec in fraud_vecs.items():
        scores[key] = float(_cosine(vec, fvec))

    # Above threshold
    selected = [k for k, s in scores.items() if s >= sim_threshold]

    # Ensure we keep top_k above minimum floor
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for k, s in top[:top_k]:
        if s >= min_sim_floor and k not in selected:
            selected.append(k)

    selected = list(dict.fromkeys(selected))  # dedupe, keep order
    return selected, scores


def _cosine(v1: List[float], v2: List[float]) -> float:
    import numpy as np

    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    dot = float((a * b).sum())
    n1 = float(np.linalg.norm(a))
    n2 = float(np.linalg.norm(b))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(dot / (n1 * n2))


# ---------------------------- DOCUMENT AGGREGATION ---------------------------


def aggregate_document_level(
    article_name: str,
    chunks: List[Chunk],
    fraud_annos: List[Dict[str, Any]],
    chunk_enrichments: List[Dict[str, Any]],
) -> Tuple[DocSemanticSummary, DocDetailedStats, List[SemanticChunk]]:
    """
    Aggregate all semantic info for a single document into:
      - doc-level summary
      - doc-level detailed stats
      - list of SemanticChunk objects
    """
    # Count mentions per fraud type
    mention_counts: Counter = Counter()
    score_sum: Counter = Counter()
    high_signal_ids: List[Tuple[int, int]] = []

    sem_chunks: List[SemanticChunk] = []

    for idx, (chunk, fraud_info, enrich_info) in enumerate(
        zip(chunks, fraud_annos, chunk_enrichments)
    ):
        # Unpack enrich info
        role = enrich_info["chunk_role"]
        is_high = enrich_info["is_high_signal"]
        ent = enrich_info["entities"]

        f_types = fraud_info["fraud_types_semantic"]
        f_scores = fraud_info["fraud_scores_semantic"]

        if is_high and f_types:
            high_signal_ids.append((chunk.page_number, idx))

        # Count and score, weighted by role
        role_weight = 1.0
        if role in ("typology_description", "red_flag_indicator"):
            role_weight = 1.5
        elif role in ("sar_guidance", "regulatory_requirement"):
            role_weight = 1.2

        for ft in f_types:
            mention_counts[ft] += 1
            score_sum[ft] += role_weight * float(f_scores.get(ft, 0.0))

        sem_chunks.append(
            SemanticChunk(
                article_name=article_name,
                page_number=chunk.page_number,
                chunk_index=idx,
                bbox=chunk.bbox,
                text=chunk.text,
                chunk_role=role,
                is_high_signal=is_high,
                fraud_types_semantic=f_types,
                fraud_scores_semantic=f_scores,
                countries_mentioned=ent["countries"],
                sectors_mentioned=ent["sectors"],
                channels_mentioned=ent["channels"],
                programs_mentioned=ent["programs"],
                actors_mentioned=ent["actors"],
            )
        )

    if mention_counts:
        sorted_scores = sorted(score_sum.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_scores[0][0]
        primary_score = sorted_scores[0][1]
        secondary = [
            k for k, v in sorted_scores[1:]
            if v >= 0.4 * primary_score
        ]
    else:
        primary = ""
        secondary = []

    summary = DocSemanticSummary(
        article_name=article_name,
        primary_fraud_type_semantic=primary,
        secondary_fraud_types_semantic=" | ".join(secondary) if secondary else "",
        semantic_mention_counts_json=json.dumps(mention_counts, ensure_ascii=False),
        total_chunks=len(chunks),
        high_signal_chunks=len(high_signal_ids),
    )

    details = DocDetailedStats(
        article_name=article_name,
        fraud_type_scores={k: float(v) for k, v in score_sum.items()},
        semantic_mention_counts={k: int(v) for k, v in mention_counts.items()},
        high_signal_chunk_ids=high_signal_ids,
    )

    return summary, details, sem_chunks


# ------------------------- METADATA MERGING HELPERS --------------------------


def load_existing_mapping(out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv)
    return pd.DataFrame()


def load_metadata_index(meta_csvs: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index {article_name -> metadata_dict} from crawler CSVs.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for csv_path in meta_csvs:
        p = Path(csv_path)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # we assume these columns *might* exist
        col_article = None
        for cand in ["article_name", "pdf_name", "file_name", "filename"]:
            if cand in df.columns:
                col_article = cand
                break
        if col_article is None:
            continue

        for _, row in df.iterrows():
            name = str(row[col_article]).strip()
            if not name:
                continue
            index[name] = {
                "fincen_id": row.get("fincen_id"),
                "doc_title": row.get("doc_title") or row.get("title"),
                "doc_date": row.get("doc_date") or row.get("date"),
                "doc_type": row.get("doc_type") or row.get("publication_type"),
                "date": row.get("date"),
                "pdf_url": row.get("pdf_url"),
                "publication_type": row.get("publication_type"),
            }
    return index


# ------------------------- KEYWORD LOCATIONS (FOR UI) ------------------------


def build_semantic_keyword_locations(
    article_name: str,
    fincen_id: Optional[str],
    sem_chunks: List[SemanticChunk],
) -> List[Dict[str, Any]]:
    """
    Produce a row per (chunk, fraud_type) to support PDF highlighting in the UI.
    """
    rows: List[Dict[str, Any]] = []
    for ch in sem_chunks:
        if not ch.is_high_signal or not ch.fraud_types_semantic:
            continue
        for ft in ch.fraud_types_semantic:
            rows.append(
                {
                    "article_name": article_name,
                    "fincen_id": fincen_id,
                    "fraud_type": ft,
                    "page_number": ch.page_number,
                    "x0_norm": ch.bbox[0],
                    "y0_norm": ch.bbox[1],
                    "x1_norm": ch.bbox[2],
                    "y1_norm": ch.bbox[3],
                }
            )
    return rows


# ------------------------------- MAIN PIPELINE --------------------------------


def main(
    pdf_dir: str,
    out_csv: str,
    out_json: str,
    out_keyword_locations: str,
    out_semantic_chunks: str,
    force: bool = False,
    extra_pdf_dirs: Optional[List[str]] = None,
) -> None:
    base = Path(".")
    pdf_dir_path = Path(pdf_dir)
    extra_dirs = [Path(d) for d in (extra_pdf_dirs or [])]

    # Collect PDFs
    pdf_paths: List[Path] = []
    seen_names = set()
    for d in [pdf_dir_path] + extra_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.pdf"):
            if p.name not in seen_names:
                seen_names.add(p.name)
                pdf_paths.append(p)
    pdf_paths.sort(key=lambda p: p.name.lower())

    if not pdf_paths:
        print("No PDFs found. Nothing to do.")
        return

    print(f"[*] Found {len(pdf_paths)} PDFs to process.")

    out_csv_path = base / out_csv
    out_json_path = base / out_json
    out_kw_path = base / out_keyword_locations
    out_sem_path = base / out_semantic_chunks

    existing_mapping = load_existing_mapping(out_csv_path)
    existing_articles = set(existing_mapping["article_name"]) if not existing_mapping.empty else set()
    print(f"[*] Existing mapping rows: {len(existing_articles)}")

    # Load existing details JSON if present
    existing_details: Dict[str, Any] = {}
    if out_json_path.exists():
        try:
            existing_details = json.loads(out_json_path.read_text(encoding="utf-8"))
        except Exception:
            existing_details = {}

    # Metadata index (titles, dates, fincen_id, etc.)
    meta_index = load_metadata_index(DEFAULT_META_CSVS)

    # Embedding function + fraud vectors
    cache_path = base / "embed_cache.json"
    embed_fn = _get_cached_embed_fn("all-MiniLM-L6-v2", cache_path)
    fraud_vecs = build_fraud_type_vectors(embed_fn)

    # Aggregation across all docs
    all_doc_summaries: List[DocSemanticSummary] = []
    all_doc_details: Dict[str, Any] = dict(existing_details)
    all_semantic_chunks: List[SemanticChunk] = []
    all_kw_rows: List[Dict[str, Any]] = []

    total = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, start=1):
        article_name = pdf_path.name
        print(f"[*] [{i}/{total}] {article_name!r}…")

        if (not force) and article_name in existing_articles and article_name in existing_details:
            print("    -> Already processed; skipping (use --force to redo).")
            continue

        chunks = extract_chunks_from_pdf(pdf_path)
        if not chunks:
            print("    -> No text chunks extracted; skipping.")
            continue

        fraud_annos: List[Dict[str, Any]] = []
        enrich_annos: List[Dict[str, Any]] = []

        for ch in chunks:
            role = classify_chunk_role(ch.text)
            ents = extract_entities_for_chunk(ch.text)
            f_types, f_scores = tag_chunk_with_fraud_types(
                ch.text, fraud_vecs, embed_fn
            )
            is_high = bool(f_types) and role in {
                "typology_description",
                "red_flag_indicator",
                "sar_guidance",
                "regulatory_requirement",
                "case_example",
            }
            fraud_annos.append(
                {
                    "fraud_types_semantic": f_types,
                    "fraud_scores_semantic": f_scores,
                }
            )
            enrich_annos.append(
                {
                    "chunk_role": role,
                    "is_high_signal": is_high,
                    "entities": ents,
                }
            )

        doc_summary, doc_details, sem_chunks = aggregate_document_level(
            article_name, chunks, fraud_annos, enrich_annos
        )

        all_doc_summaries.append(doc_summary)
        all_doc_details[article_name] = asdict(doc_details)
        all_semantic_chunks.extend(sem_chunks)

        meta = meta_index.get(article_name, {})
        fincen_id = meta.get("fincen_id")
        kw_rows = build_semantic_keyword_locations(article_name, fincen_id, sem_chunks)
        all_kw_rows.extend(kw_rows)

        if i % 5 == 0:
            print("    -> Saving intermediate details JSON…")
            out_json_path.write_text(
                json.dumps(all_doc_details, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Merge doc summaries into mapping CSV
    new_mapping_df = pd.DataFrame([asdict(s) for s in all_doc_summaries])
    if not existing_mapping.empty:
        merged = existing_mapping.merge(
            new_mapping_df,
            on="article_name",
            how="outer",
            suffixes=("", "_new"),
        )
        for col in new_mapping_df.columns:
            if col == "article_name":
                continue
            new_col = f"{col}_new"
            if new_col in merged.columns:
                merged[col] = merged[new_col].combine_first(merged[col])
                merged.drop(columns=[new_col], inplace=True, errors="ignore")
        final_mapping = merged
    else:
        final_mapping = new_mapping_df

    final_mapping.to_csv(out_csv_path, index=False)
    print(f"[*] Wrote mapping CSV: {out_csv_path}")

    out_json_path.write_text(
        json.dumps(all_doc_details, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[*] Wrote details JSON: {out_json_path}")

    sem_rows: List[Dict[str, Any]] = []
    for ch in all_semantic_chunks:
        sem_rows.append(
            {
                "article_name": ch.article_name,
                "page_number": ch.page_number,
                "chunk_index": ch.chunk_index,
                "x0_norm": ch.bbox[0],
                "y0_norm": ch.bbox[1],
                "x1_norm": ch.bbox[2],
                "y1_norm": ch.bbox[3],
                "text": ch.text,
                "chunk_role": ch.chunk_role,
                "is_high_signal": ch.is_high_signal,
                "fraud_types_semantic": " | ".join(ch.fraud_types_semantic),
                "fraud_scores_semantic": json.dumps(ch.fraud_scores_semantic),
                "countries_mentioned": " | ".join(ch.countries_mentioned),
                "sectors_mentioned": " | ".join(ch.sectors_mentioned),
                "channels_mentioned": " | ".join(ch.channels_mentioned),
                "programs_mentioned": " | ".join(ch.programs_mentioned),
                "actors_mentioned": " | ".join(ch.actors_mentioned),
            }
        )

    pd.DataFrame(sem_rows).to_csv(out_sem_path, index=False)
    print(f"[*] Wrote semantic chunks CSV: {out_sem_path}")

    if all_kw_rows:
        kw_fieldnames = [
            "article_name",
            "fincen_id",
            "fraud_type",
            "page_number",
            "x0_norm",
            "y0_norm",
            "x1_norm",
            "y1_norm",
        ]
        with out_kw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=kw_fieldnames)
            writer.writeheader()
            for row in all_kw_rows:
                writer.writerow(row)
        print(f"[*] Wrote semantic highlight CSV: {out_kw_path}")
    else:
        print("[*] No high-signal fraud-type chunks found; skipping highlight CSV.")


# ------------------------------- CLI PARSING ---------------------------------


def _parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(
        description="Semantic fraud mapping over FinCEN PDFs."
    )
    p.add_argument("pdf_dir", help="Main directory containing FinCEN PDFs.")
    p.add_argument("out_csv", help=f"Document-level mapping CSV (default: {OUT_DOC_CSV}).")
    p.add_argument("out_json", help=f"Details JSON (default: {OUT_DETAILS_JSON}).")
    p.add_argument(
        "out_keywords",
        help=f"Semantic highlight CSV (replacement for keyword locations; default: {OUT_KEYWORD_LOCATIONS}).",
    )
    p.add_argument(
        "out_semantic",
        help=f"Chunk-level semantic CSV (default: {OUT_SEMANTIC_CHUNKS}).",
    )
    p.add_argument(
        "--extra-pdf-dir",
        action="append",
        default=[],
        help="Additional directory with FinCEN PDFs (e.g., alerts, notices). May be used multiple times.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess PDFs even if already present in mapping & details JSON.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    args = _parse_args()
    main(
        pdf_dir=args.pdf_dir,
        out_csv=args.out_csv,
        out_json=args.out_json,
        out_keyword_locations=args.out_keywords,
        out_semantic_chunks=args.out_semantic,
        force=args.force,
        extra_pdf_dirs=args.extra_pdf_dir,
    )
