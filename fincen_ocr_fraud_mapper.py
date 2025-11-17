#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run as:  uv run python fincen_ocr_fraud_mapper.py `
  fincen_advisory_pdfs `
  fincen_fraud_mapping.csv `
  fincen_fraud_mapping_details.json `
  fincen_keyword_locations.csv `
  fincen_semantic_chunks.csv `
  --extra-pdf-dir fincen_alert_pdfs `
  --extra-pdf-dir fincen_notice_pdfs


fincen_ocr_fraud_mapper.py (UNIFIED)
---------------------------------
End-to-end parser for FINCEN PDFs (Advisories, Alerts, Notices) that:

  1) Walks one or more folders of PDFs (advisories, alerts, notices, etc.)
  2) Extracts text as CHUNKS with page geometry (page_number, bbox)
        - PyMuPDF direct text blocks when available
        - PaddleOCR (or Tesseract) when the PDF is image-only
  3) Tags each chunk semantically with fraud types using FREE local embeddings
  4) Records REGEX keyword hits with precise page/bbox locations
  5) Joins in DOC-LEVEL METADATA from your crawler CSVs:
        - fincen_advisories.csv
        - fincen_alerts.csv        (if present)
        - fincen_notices.csv       (if present)
     and attaches:
        - fincen_id
        - doc_type (Advisory / Alert / Notice)
        - doc_title
        - doc_date
        - source_csv
  6) Writes four artifacts for analytics and highlighting:
        - fincen_fraud_mapping.csv          (PDF-level summary, all doc types)
        - fincen_fraud_mapping_details.json (placeholder for future rich details)
        - fincen_keyword_locations.csv      (exact keyword hits + locations)
        - fincen_semantic_chunks.csv        (per-chunk semantic tags + sims)

CLI EXAMPLES
------------
  # Process only advisories (as before)
  uv run python fincen_ocr_fraud_mapper.py fincen_advisory_pdfs \
      fincen_fraud_mapping.csv fincen_fraud_mapping_details.json \
      fincen_keyword_locations.csv fincen_semantic_chunks.csv

  # Process advisories + alerts + notices in ONE unified pass
  uv run python fincen_ocr_fraud_mapper.py fincen_advisory_pdfs \
      fincen_fraud_mapping.csv fincen_fraud_mapping_details.json \
      fincen_keyword_locations.csv fincen_semantic_chunks.csv \
      --extra-pdf-dir fincen_alert_pdfs \
      --extra-pdf-dir fincen_notice_pdfs

DESIGN NOTES (ruthless version)
-------------------------------
- We STOP pretending each artifact is only about “advisories”. Everything is now
  one cross-document, cross-type fraud library.
- We JOIN on pdf_filename so every row (summary, chunk, keyword) knows:
    * which FIN- ID it belongs to,
    * what type (Advisory/Alert/Notice),
    * what title + date it came from.
- Supabase is deliberately removed from this script. You can add a dedicated
  "db_sync" step later instead of bloating the mapper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

# -------------------------- CONFIG (override via CLI) --------------------------

# NOTE: pdf_dir + --extra-pdf-dir control WHAT you process.
# You can still run just advisories; when you're serious, pass alerts + notices too.

PDF_FOLDER = "fincen_advisory_pdfs"  # default positional argument

OUT_CSV = "fincen_fraud_mapping.csv"
OUT_JSON = "fincen_fraud_mapping_details.json"

OUT_KEYWORD_LOCATIONS = "fincen_keyword_locations.csv"   # article, fraud type, keyword, location
OUT_SEMANTIC_CHUNKS   = "fincen_semantic_chunks.csv"     # per-chunk semantic tagging (page + bbox)

# Embeddings (no API keys needed)
SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Semantic tagging knobs (tune on a few PDFs)
SIM_THRESHOLD   = 0.42   # primary cutoff
TOP_K           = 2      # always keep top-K labels per chunk (if above min floor)
MIN_SIM_FLOOR   = 0.35
MAX_CHUNK_CHARS = 1400   # safety to avoid huge paragraphs per chunk

# OCR rendering for image-based PDFs
OCR_SCALE = 2.75  # 2.5–3.0 is a good range for Paddle/Tesseract

# Caching to speed re-runs
STATE_FILE = "processing_state.json"  # remembers PDF sha256 -> processed
EMBED_CACHE = "embed_cache.json"      # remembers text_hash -> embedding

# Where to stash temporary OCR images
TMP_DIR = Path("fincen_ocr_tmp")

# Metadata sources we'll try to join in automatically
METADATA_FILES = [
    "fincen_advisories.csv",  # advisories crawler
    "fincen_alerts.csv",      # alerts crawler (your new one)
    "fincen_notices.csv",     # notices crawler (your new one)
]

# -------------------------------- Taxonomy ------------------------------------

FRAUD_TYPES: Dict[str, List[str]] = {
    "money_laundering": [
        "money laundering and AML indicators",
        "placement layering integration",
        "suspicious activity reports SAR",
        "beneficial ownership shell companies",
    ],
    "structuring_smurfing": [
        "structuring deposits under CTR threshold",
        "smurfing cash deposits to avoid reporting",
        "under-reporting patterns",
    ],
    "terrorist_financing": [
        "terrorism financing TF",
        "sanctions designated persons OFAC lists",
        "funds moving to extremist groups",
    ],
    "human_trafficking": [
        "human trafficking HT",
        "sex trafficking forced labor",
        "movement of funds tied to exploitation",
    ],
    "ransomware": [
        "ransomware double extortion ransom payment",
        "RaaS crypto wallets and mixers",
    ],
    "bec_business_email_compromise": [
        "business email compromise BEC",
        "spoofing impersonation fraudulent wire transfer",
    ],
    "check_fraud": [
        "check fraud altered or washed checks",
        "forged check activity",
    ],
    "romance_scam_elder_fraud": [
        "romance scam sweetheart scam",
        "elder fraud social engineering",
    ],
    "synthetic_identity": [
        "synthetic identity fraud",
        "mismatched SSN credit profile numbers",
    ],
    "crypto_virtual_assets": [
        "virtual assets crypto exchanges mixers tumblers",
        "DeFi blockchain stablecoins",
    ],
    "trade_based_money_laundering": [
        "trade based money laundering TBML",
        "over-invoicing under-invoicing trade misinvoicing",
    ],
    "sanctions_evasion": [
        "sanctions evasion circumvention",
        "front companies third-country routing",
    ],
    "corruption_bribery": [
        "foreign corrupt practices bribery kickbacks",
        "politically exposed persons PEP",
    ],
    "mortgage_financial_aid_fraud": [
        "mortgage fraud",
        "PPP EIDL CARES Act fraudulent loans",
    ],
    "healthcare_fraud": [
        "health care fraud medical billing upcoding",
        "unbundling fraudulent claims",
    ],
    "tax_evasion": [
        "tax evasion false returns undisclosed income",
    ],
}

FRAUD_KEYWORDS: Dict[str, List[str]] = {
    "money_laundering": [
        r"\bmoney laundering\b",
        r"\bplacement\b", r"\blayering\b", r"\bintegration\b",
        r"\bAML\b", r"\bBSA\b", r"\bSARs?\b",
        r"\bbeneficial owner(ship)?\b",
        r"\bshell compan(y|ies)\b",
    ],
    "structuring_smurfing": [
        r"\bstructuring\b", r"\bsmurf(ing|ers?)\b",
        r"\bCTR(s)?\b", r"\b\$?9,?000\b", r"\bund(er|)-reporting\b",
    ],
    "terrorist_financing": [
        r"\bterror(ist|ism) financing\b", r"\bTF\b", r"\bOFAC\b",
        r"\bsanctions?\b", r"\bdesignated persons?\b",
    ],
    "human_trafficking": [
        r"\bhuman trafficking\b", r"\bHT\b", r"\bsex trafficking\b", r"\bforced labor\b",
    ],
    "ransomware": [
        r"\bransomware\b", r"\bdouble extortion\b", r"\bRaaS\b", r"\bransom\b",
    ],
    "bec_business_email_compromise": [
        r"\bB(\.| )?E(\.| )?C\b", r"\bbusiness email compromise\b",
        r"\bfraudulent wire\b", r"\bspoof(ed|ing)\b", r"\bimpersonation\b",
    ],
    "check_fraud": [
        r"\bcheck fraud\b", r"\baltered checks?\b",
        r"\bforged( |-)checks?\b", r"\bwash(ed|ing) checks?\b",
    ],
    "romance_scam_elder_fraud": [
        r"\bromance scam(s)?\b", r"\belder(?:ly)? fraud\b", r"\bsweetheart scam\b",
    ],
    "synthetic_identity": [
        r"\bsynthetic identit(y|ies)\b", r"\bC ?P ?N\b", r"\bSSN (mismatch|mismatched)\b",
    ],
    "crypto_virtual_assets": [
        r"\bvirtual asset(s)?\b", r"\bVASP(s)?\b", r"\bexchange wallet\b",
        r"\bmixer(s)?\b", r"\btumbler(s)?\b", r"\bblockchain\b", r"\bDeFi\b",
        r"\bstablecoin(s)?\b", r"\bcrypto(currency)?\b",
    ],
    "trade_based_money_laundering": [
        r"\bTBML\b", r"\btrade-?based money laundering\b",
        r"\bover-?invoicing\b", r"\bunder-?invoicing\b",
    ],
    "sanctions_evasion": [
        r"\bsanctions evasion\b", r"\bcircumvention\b", r"\bfront\b",
        r"\bthird-?country routing\b",
    ],
    "corruption_bribery": [
        r"\bFCPA\b", r"\bbriber(y|ies)\b", r"\bcorruption\b", r"\bkickbacks?\b", r"\bPEPs?\b",
    ],
    "mortgage_financial_aid_fraud": [
        r"\bmortgage fraud\b", r"\bPPP\b", r"\bEIDL\b", r"\bCARES Act\b",
        r"\bloan (forgiveness|forgiven)\b",
    ],
    "healthcare_fraud": [
        r"\bhealth care fraud\b", r"\bmedical billing\b", r"\bupcoding\b", r"\bunbundling\b",
    ],
    "tax_evasion": [
        r"\btax evasion\b", r"\bfalse returns?\b", r"\bundisclosed income\b",
    ],
}

# ------------------------------- Logging --------------------------------------

LOG_FORMAT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("fincen")


# ------------------------------- Utilities ------------------------------------

def compile_patterns(mapping: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """Pre-compile regex patterns for speed + correctness."""
    return {label: [re.compile(p, re.IGNORECASE) for p in pats] for label, pats in mapping.items()}


@dataclass
class Chunk:
    """
    A single text chunk with geometry.
    We keep absolute and normalized bboxes so the front-end can draw rectangles
    reliably at any viewer scale.
    """
    file: str
    page_number: int  # 1-based
    bbox: Tuple[float, float, float, float]  # absolute [x0, y0, x1, y1]
    page_width: float
    page_height: float
    text: str

    @property
    def bbox_norm(self) -> Tuple[float, float, float, float]:
        if self.page_width <= 0 or self.page_height <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        x0, y0, x1, y1 = self.bbox
        return (
            x0 / self.page_width,
            y0 / self.page_height,
            x1 / self.page_width,
            y1 / self.page_height,
        )


def _merge_bbox(
    b1: Optional[Tuple[float, float, float, float]],
    b2: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Union of two rectangles; used for simple line->paragraph grouping in OCR."""
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)


def _clip_text(s: str, maxlen: int = MAX_CHUNK_CHARS) -> str:
    """Hard cap chunk length so embeddings stay fast & consistent."""
    s = s.strip()
    return (s[: maxlen].rstrip() + " …") if len(s) > maxlen else s


# ------------------------- PDF → chunks (direct text) -------------------------

def extract_chunks_direct(pdf_path: Path) -> List[Chunk]:
    """Use PyMuPDF text blocks with bbox (fast, keeps exact geometry)."""
    doc = fitz.open(pdf_path)
    out: List[Chunk] = []
    for pi in range(len(doc)):
        page = doc[pi]
        W, H = float(page.rect.width), float(page.rect.height)
        blocks = page.get_text("blocks")

        merged: List[Tuple[Tuple[float, float, float, float], str]] = []
        for blk in blocks:
            if len(blk) < 5:
                continue
            x0, y0, x1, y1, txt = float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]), str(blk[4])
            text = " ".join(line.strip() for line in txt.splitlines()).strip()
            if not text:
                continue
            merged.append(((x0, y0, x1, y1), text))

        # Sort and greedy-merge by proximity
        merged.sort(key=lambda t: (t[0][1], t[0][0]))
        current_bbox = None
        current_text: List[str] = []
        last_y = None

        for bb, tx in merged:
            if last_y is None or abs(bb[1] - last_y) < 10:  # vertical gap threshold
                current_bbox = _merge_bbox(current_bbox, bb)
                current_text.append(tx)
            else:
                if current_text and current_bbox is not None:
                    out.append(
                        Chunk(
                            str(pdf_path),
                            pi + 1,
                            current_bbox,
                            W,
                            H,
                            _clip_text(" ".join(current_text)),
                        )
                    )
                current_bbox = bb
                current_text = [tx]
            last_y = bb[1]

        if current_text and current_bbox is not None:
            out.append(
                Chunk(
                    str(pdf_path),
                    pi + 1,
                    current_bbox,
                    W,
                    H,
                    _clip_text(" ".join(current_text)),
                )
            )
    doc.close()
    return out


# ------------------------ Image rendering for OCR path ------------------------

def pdf_to_images(pdf_path: Path, out_dir: Path, scale: float = OCR_SCALE) -> List[Tuple[Path, int, int]]:
    """Render pages to PNG for OCR. Returns list of tuples: (image_path, width, height)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_meta: List[Tuple[Path, int, int]] = []
    try:
        from PIL import Image as PILImage  # noqa: F401
        have_pil = True
    except Exception:
        have_pil = False

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        p = out_dir / f"{pdf_path.stem}_page_{i+1:04d}.png"
        if have_pil:
            from PIL import Image as PILImage
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(p)
        else:
            pix.save(str(p))
        img_meta.append((p, pix.width, pix.height))
    doc.close()
    return img_meta


# ---------------------- OCR (Paddle first; Tesseract else) --------------------

def ocr_images_to_chunks(image_meta: List[Tuple[Path, int, int]], pdf_file: str) -> List[Chunk]:
    """
    Run OCR and group lines into paragraph-ish chunks.
    Keeps absolute bbox + page dimensions for highlight fidelity.
    """
    # Try PaddleOCR
    use_paddle = False
    try:
        import importlib
        mod = importlib.import_module("paddleocr")
        PaddleOCR = getattr(mod, "PaddleOCR", None)
        if PaddleOCR is not None:
            use_paddle = True
    except Exception:
        use_paddle = False

    chunks: List[Chunk] = []

    if use_paddle:
        ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        for idx, (img_path, W, H) in enumerate(image_meta, start=1):
            res = ocr.ocr(str(img_path), cls=True)
            lines: List[Tuple[Tuple[float, float, float, float], str]] = []
            for line in res:
                for det in line:
                    bbox = det[0]  # 4 points
                    text = det[1][0]
                    if not text or not text.strip():
                        continue
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    lines.append(((x0, y0, x1, y1), text.strip()))

            # group lines vertically
            lines.sort(key=lambda t: (t[0][1], t[0][0]))
            group: List[Tuple[Tuple[float, float, float, float], str]] = []
            last_y = None
            for bb, tx in lines:
                if last_y is None or abs(bb[1] - last_y) < 16:
                    group.append((bb, tx))
                else:
                    if group:
                        text = _clip_text(" ".join(t for _, t in group))
                        bbox = None
                        for b, _ in group:
                            bbox = _merge_bbox(bbox, b)
                        if bbox is not None:
                            chunks.append(
                                Chunk(pdf_file, idx, bbox, float(W), float(H), text)
                            )
                    group = [(bb, tx)]
                last_y = bb[1]
            if group:
                text = _clip_text(" ".join(t for _, t in group))
                bbox = None
                for b, _ in group:
                    bbox = _merge_bbox(bbox, b)
                if bbox is not None:
                    chunks.append(
                        Chunk(pdf_file, idx, bbox, float(W), float(H), text)
                    )
    else:
        # Fallback to pytesseract with TSV to preserve coordinates
        try:
            from PIL import Image as PILImage
            import pytesseract
        except Exception as e:
            raise RuntimeError(
                "OCR backend not available: install 'paddleocr' (and paddlepaddle) "
                "or install 'pytesseract' with Pillow."
            ) from e

        for idx, (img_path, W, H) in enumerate(image_meta, start=1):
            img = PILImage.open(img_path)
            tsv = pytesseract.image_to_data(
                img, lang="eng", output_type=pytesseract.Output.DATAFRAME
            )
            tsv = tsv.dropna(subset=["text"])
            for (_, _, _), df in tsv.groupby(["block_num", "par_num", "line_num"]):
                df = df[df["conf"] != -1]
                if df.empty:
                    continue
                line_text = " ".join(str(x) for x in df["text"] if str(x).strip())
                if not line_text.strip():
                    continue
                x0 = float(df["left"].min())
                y0 = float(df["top"].min())
                x1 = float((df["left"] + df["width"]).max())
                y1 = float((df["top"] + df["height"]).max())
                chunks.append(
                    Chunk(
                        pdf_file,
                        idx,
                        (x0, y0, x1, y1),
                        float(W),
                        float(H),
                        _clip_text(line_text),
                    )
                )
    return chunks


def extract_chunks(pdf_path: Path, tmp_dir: Path) -> Tuple[List[Chunk], dict]:
    """Choose best path: direct text vs. OCR. Records mode + pages in meta for later QA."""
    meta = {"file": str(pdf_path), "mode": None, "pages": 0}
    doc = fitz.open(pdf_path)
    meta["pages"] = len(doc)
    text_has_content = any(doc[i].get_text("text").strip() for i in range(len(doc)))
    doc.close()
    if text_has_content:
        meta["mode"] = "direct"
        return extract_chunks_direct(pdf_path), meta
    meta["mode"] = "ocr"
    images = pdf_to_images(pdf_path, tmp_dir / f"{pdf_path.stem}_pages", scale=OCR_SCALE)
    chunks = ocr_images_to_chunks(images, str(pdf_path))
    return chunks, meta


# --------------------------- Embedding back-end -------------------------------

def get_embedder():
    """
    Returns (embed_fn, model_name, backend_tag).
    Caches vectors on disk by text hash to speed re-runs.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Install sentence-transformers for free local embeddings:\n"
            "  pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(SENTENCE_TRANSFORMERS_MODEL)

    cache_path = Path(EMBED_CACHE)
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    def _hash_text(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    def _embed(texts: List[str]) -> List[List[float]]:
        # Normalize embeddings so dot == cosine
        result: List[List[float]] = []
        to_compute: List[str] = []
        missing_idx: List[int] = []
        for i, t in enumerate(texts):
            h = _hash_text(t)
            if h in cache:
                result.append(cache[h])
            else:
                result.append([])  # placeholder
                to_compute.append(t)
                missing_idx.append(i)
        if to_compute:
            vecs = model.encode(to_compute, normalize_embeddings=True)
            for i, v in zip(missing_idx, vecs.tolist()):
                h = _hash_text(texts[i])
                result[i] = v
                cache[h] = v
            # write-through cache every batch
            try:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except Exception:
                pass
        return result

    return _embed, SENTENCE_TRANSFORMERS_MODEL, "sbert"


def cosine_sim(a: List[float], b: List[float]) -> float:
    # SBERT vectors are normalized above, so dot=cosine
    return float(sum(x * y for x, y in zip(a, b)))


def build_fraud_type_embeddings(embed_fn) -> Dict[str, List[float]]:
    type_texts = [" ".join(seeds) for _, seeds in FRAUD_TYPES.items()]
    vecs = embed_fn(type_texts)
    return {ftype: vec for ftype, vec in zip(FRAUD_TYPES.keys(), vecs)}


def tag_chunks_with_fraud_types(
    chunks: List[Chunk],
    embed_fn,
    type_vecs: Dict[str, List[float]],
    threshold: float = SIM_THRESHOLD,
    top_k: int = TOP_K,
    min_floor: float = MIN_SIM_FLOOR,
) -> List[Dict[str, Any]]:
    """Compute per-chunk similarity to each fraud type.

    Selection rule:
      - keep all types with sim >= threshold
      - also keep up to top_k types if they clear min_floor

    WHY: avoids missing near-miss chunks while limiting noise.
    """
    out: List[Dict[str, Any]] = []
    texts = [c.text for c in chunks]
    if not texts:
        return out

    chunk_vecs = embed_fn(texts)
    ftype_items = list(type_vecs.items())

    for c, v in zip(chunks, chunk_vecs):
        sims = {name: cosine_sim(v, vec) for name, vec in ftype_items}
        # keep by threshold
        matches = [name for name, s in sims.items() if s >= threshold]
        # also keep top-k above floor
        if len(matches) < top_k:
            top_pairs = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            for name, s in top_pairs:
                if name not in matches and s >= min_floor:
                    matches.append(name)
                if len(matches) >= top_k:
                    break
        out.append(
            {
                "file": c.file,
                "page_number": c.page_number,
                "page_width": c.page_width,
                "page_height": c.page_height,
                "bbox": [c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3]],
                "bbox_norm": list(c.bbox_norm),
                "text": c.text,
                "matched_fraud_types": matches,
                "similarities": sims,
            }
        )
    return out


# ----------------------------- Regex scoring ----------------------------------

def score_fraud_types_regex(
    text: str, patterns: Dict[str, List[re.Pattern]]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    counts = {label: 0 for label in patterns}
    matches = {label: [] for label in patterns}
    for label, regs in patterns.items():
        for rgx in regs:
            for m in rgx.finditer(text):
                counts[label] += 1
                snippet = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                matches[label].append(snippet)

    # de-duplicate
    for label in matches:
        seen = set()
        uniq = []
        for s in matches[label]:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        matches[label] = uniq[:20]
    return counts, matches


# ------------------ Exact keyword hits WITH locations -------------------------

def keyword_hits_with_locations(
    chunks: List[Chunk],
    compiled_patterns: Dict[str, List[re.Pattern]],
) -> List[Dict[str, Any]]:
    """Returns rows with precise locations for each regex hit."""
    rows = []
    for ch in chunks:
        article_name = Path(ch.file).name
        for fraud_type, regs in compiled_patterns.items():
            for rgx in regs:
                for m in rgx.finditer(ch.text):
                    rows.append(
                        {
                            "article_name": article_name,
                            "fraud_type": fraud_type,
                            "fraud_keyword": m.group(0),
                            "page_number": ch.page_number,
                            "bbox": json.dumps(
                                [ch.bbox[0], ch.bbox[1], ch.bbox[2], ch.bbox[3]]
                            ),
                            "bbox_norm": json.dumps(list(ch.bbox_norm)),
                            "page_width": ch.page_width,
                            "page_height": ch.page_height,
                        }
                    )
    return rows


# --------------------------- Hash + state helpers -----------------------------

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict[str, Any]:
    if Path(STATE_FILE).exists():
        try:
            return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    try:
        Path(STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------- Metadata join helpers ----------------------------

def _guess_doc_type_from_filename(csv_name: str) -> Optional[str]:
    low = csv_name.lower()
    if "advis" in low:
        return "Advisory"
    if "alert" in low:
        return "Alert"
    if "notice" in low or "ntc" in low:
        return "Notice"
    return None


def _parse_fincen_id(text: str) -> str:
    """
    Try to pull a FIN- style code out of a title or description.
    Examples: FIN-2024-A001, FIN-2022-A002, FIN-2023-Alert001, FIN-2024-NTC2, etc.
    """
    if not text:
        return ""
    m = re.search(r"FIN-\d{4}-[A-Za-z0-9]+", text)
    return m.group(0).upper() if m else ""


def load_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from known CSVs and build a mapping:
        pdf_filename -> { fincen_id, doc_type, doc_title, doc_date, source_csv }
    If we don't find metadata for a PDF, that's fine — it just gets blank fields.
    """
    meta_by_pdf: Dict[str, Dict[str, Any]] = {}

    for fname in METADATA_FILES:
        path = Path(fname)
        if not path.exists():
            continue

        logger.info(f"Loading metadata from {fname}")
        default_type = _guess_doc_type_from_filename(fname)

        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdf_filename = (row.get("pdf_filename") or "").strip()
                    if not pdf_filename:
                        continue

                    # If we've already seen this pdf_filename from a "richer" CSV, don't override
                    if pdf_filename in meta_by_pdf:
                        continue

                    title = (row.get("title") or "").strip()
                    date = (row.get("date") or "").strip()
                    doc_type = (row.get("doc_type") or "").strip() or default_type or ""
                    fincen_id = (row.get("fincen_id") or "").strip()
                    if not fincen_id:
                        fincen_id = _parse_fincen_id(title)

                    meta_by_pdf[pdf_filename] = {
                        "pdf_filename": pdf_filename,
                        "fincen_id": fincen_id,
                        "doc_type": doc_type,
                        "doc_title": title,
                        "doc_date": date,
                        "source_csv": fname,
                    }
        except Exception as e:
            logger.warning(f"Failed to load metadata from {fname}: {e}")

    if not meta_by_pdf:
        logger.warning(
            "No metadata loaded. "
            "You will still get fraud mapping, but rows won't have fincen_id/doc_type/title/date."
        )
    else:
        logger.info(f"Metadata entries loaded: {len(meta_by_pdf)}")

    return meta_by_pdf


def attach_meta_to_row(
    row: Dict[str, Any],
    filename: str,
    meta_by_pdf: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Add fincen_id/doc_type/title/date/source_csv to a row if we know them."""
    meta = meta_by_pdf.get(filename)
    if not meta:
        # still add explicit empty fields so downstream schema is stable
        row.setdefault("article_name", filename)
        row.setdefault("fincen_id", "")
        row.setdefault("doc_type", "")
        row.setdefault("doc_title", "")
        row.setdefault("doc_date", "")
        row.setdefault("source_csv", "")
        return row

    row.setdefault("article_name", filename)
    row.setdefault("fincen_id", meta.get("fincen_id", ""))
    row.setdefault("doc_type", meta.get("doc_type", ""))
    row.setdefault("doc_title", meta.get("doc_title", ""))
    row.setdefault("doc_date", meta.get("doc_date", ""))
    row.setdefault("source_csv", meta.get("source_csv", ""))
    return row


# ------------------------------- CSV helpers ----------------------------------

def _write_csv(
    path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None
) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ------------------------------- Main pipeline --------------------------------

def process_pdf(
    pdf: Path,
    tmp_dir: Path,
    embed_fn,
    type_vecs: Dict[str, List[float]],
    compiled_patterns: Dict[str, List[re.Pattern]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single PDF and return (summary_row, semantic_rows, keyword_rows)."""
    chunks, meta = extract_chunks(pdf, tmp_dir)

    # regex summary (per PDF)
    full_text = "\n".join(c.text for c in chunks)
    counts, _matched_snips = score_fraud_types_regex(full_text, compiled_patterns)
    total_hits = sum(counts.values())
    top = sorted(
        [(lbl, c) for lbl, c in counts.items() if c > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    top_labels = [f"{lbl}:{cnt}" for lbl, cnt in top]

    # semantic tagging per chunk (for highlight-by-topic)
    tagged = tag_chunks_with_fraud_types(
        chunks, embed_fn, type_vecs, threshold=SIM_THRESHOLD, top_k=TOP_K, min_floor=MIN_SIM_FLOOR
    )

    # exact keyword hits WITH locations
    kw_rows = keyword_hits_with_locations(chunks, compiled_patterns)

    # flatten semantic rows for CSV
    semantic_rows: List[Dict[str, Any]] = []
    for t in tagged:
        semantic_rows.append(
            {
                "article_name": Path(t["file"]).name,
                "page_number": t["page_number"],
                "page_width": t["page_width"],
                "page_height": t["page_height"],
                "bbox": json.dumps(t["bbox"]),
                "bbox_norm": json.dumps(t["bbox_norm"]),
                "text": t["text"],
                "matched_fraud_types": ",".join(t["matched_fraud_types"]),
                "similarities_json": json.dumps(t["similarities"]),
                "embedding_model": SENTENCE_TRANSFORMERS_MODEL,
                "sim_threshold": SIM_THRESHOLD,
                "top_k": TOP_K,
                "min_sim_floor": MIN_SIM_FLOOR,
            }
        )

    summary_row = {
        "file": str(pdf),
        "article_name": pdf.name,
        "sha256": file_sha256(pdf),
        "mode": meta["mode"],
        "pages": meta["pages"],
        "total_hits_regex": total_hits,
        "top_labels_regex": "; ".join(top_labels),
        "embedding_backend": "sbert",
        "embedding_model": SENTENCE_TRANSFORMERS_MODEL,
        "sim_threshold": SIM_THRESHOLD,
        "top_k": TOP_K,
        "min_sim_floor": MIN_SIM_FLOOR,
    }

    return summary_row, semantic_rows, kw_rows


def main(
    pdf_dir: str = PDF_FOLDER,
    out_csv: str = OUT_CSV,
    out_json: str = OUT_JSON,
    out_keyword_locations: str = OUT_KEYWORD_LOCATIONS,
    out_semantic_chunks: str = OUT_SEMANTIC_CHUNKS,
    force: bool = False,
    extra_pdf_dirs: Optional[List[str]] = None,
) -> None:
    # Resolve PDF directories
    dirs: List[Path] = []
    base = Path(pdf_dir)
    if base.exists():
        dirs.append(base)
    else:
        logger.warning(f"PDF folder not found: {pdf_dir}")

    extra_pdf_dirs = extra_pdf_dirs or []
    for d in extra_pdf_dirs:
        p = Path(d)
        if p.exists():
            dirs.append(p)
        else:
            logger.warning(f"Extra PDF folder not found: {d}")

    if not dirs:
        raise FileNotFoundError("No valid PDF folders supplied.")

    TMP_DIR.mkdir(exist_ok=True)

    # Embeddings + regex
    t0 = time.time()
    embed_fn, embed_model, backend = get_embedder()
    type_vecs = build_fraud_type_embeddings(embed_fn)
    compiled = compile_patterns(FRAUD_KEYWORDS)

    # Metadata
    meta_by_pdf = load_metadata()

    # State for incremental runs
    state = load_state()
    state.setdefault("pdf_hashes", {})
    processed_hashes = state["pdf_hashes"]

    summary_rows: List[Dict[str, Any]] = []
    details_bundle: List[Dict[str, Any]] = []  # kept for future expansion if needed
    keyword_loc_rows: List[Dict[str, Any]] = []
    semantic_chunk_rows: List[Dict[str, Any]] = []

    # Gather all PDFs across folders
    pdfs: List[Path] = []
    for d in dirs:
        pdfs.extend([p for p in sorted(d.glob("**/*.pdf")) if p.is_file()])

    if not pdfs:
        logger.warning("No PDFs found in supplied folders.")
        return

    logger.info(f"Total PDFs to consider: {len(pdfs)}")

    for i, pdf in enumerate(pdfs, 1):
        pdf_hash = file_sha256(pdf)
        already = processed_hashes.get(str(pdf))

        if already == pdf_hash and not force:
            logger.info(f"[{i}/{len(pdfs)}] {pdf.name} (skip: unchanged)")
            continue

        logger.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            s_row, sem_rows, kw_rows = process_pdf(
                pdf, TMP_DIR, embed_fn, type_vecs, compiled
            )

            # Attach metadata to summary row
            s_row = attach_meta_to_row(s_row, pdf.name, meta_by_pdf)
            summary_rows.append(s_row)

            # Attach metadata to semantic rows
            for r in sem_rows:
                r = attach_meta_to_row(r, r.get("article_name", pdf.name), meta_by_pdf)
                semantic_chunk_rows.append(r)

            # Attach metadata to keyword rows
            for r in kw_rows:
                r = attach_meta_to_row(r, r.get("article_name", pdf.name), meta_by_pdf)
                keyword_loc_rows.append(r)

            # mark processed
            processed_hashes[str(pdf)] = pdf_hash
        except Exception as e:
            logger.exception(f"  !! Error on {pdf.name}: {e}")
            err_row = {
                "file": str(pdf),
                "article_name": pdf.name,
                "sha256": pdf_hash,
                "mode": "error",
                "pages": None,
                "total_hits_regex": None,
                "top_labels_regex": f"ERROR: {e}",
                "embedding_backend": backend,
                "embedding_model": embed_model,
                "sim_threshold": SIM_THRESHOLD,
                "top_k": TOP_K,
                "min_sim_floor": MIN_SIM_FLOOR,
            }
            err_row = attach_meta_to_row(err_row, pdf.name, meta_by_pdf)
            summary_rows.append(err_row)

    # write outputs
    _write_csv(
        out_csv,
        summary_rows,
        fieldnames=[
            "file",
            "article_name",
            "fincen_id",
            "doc_type",
            "doc_title",
            "doc_date",
            "source_csv",
            "sha256",
            "mode",
            "pages",
            "total_hits_regex",
            "top_labels_regex",
            "embedding_backend",
            "embedding_model",
            "sim_threshold",
            "top_k",
            "min_sim_floor",
        ],
    )

    # details JSON (placeholder for parity with earlier version)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(details_bundle, f, ensure_ascii=False, indent=2)

    _write_csv(
        out_keyword_locations,
        keyword_loc_rows,
        fieldnames=[
            "article_name",
            "fincen_id",
            "doc_type",
            "doc_title",
            "doc_date",
            "source_csv",
            "fraud_type",
            "fraud_keyword",
            "page_number",
            "bbox",
            "bbox_norm",
            "page_width",
            "page_height",
        ],
    )

    _write_csv(out_semantic_chunks, semantic_chunk_rows)

    # persist state (so unchanged PDFs get skipped next run)
    state["pdf_hashes"] = processed_hashes
    save_state(state)

    dt = time.time() - t0
    logger.info("\nDone.")
    logger.info(f"Elapsed: {dt:.1f}s")
    logger.info(f"Summary CSV: {out_csv}")
    logger.info(f"Details JSON: {out_json}")
    logger.info(f"Keyword locations CSV: {out_keyword_locations}")
    logger.info(f"Semantic chunks CSV: {out_semantic_chunks}")
    logger.info(
        "Tip: Use fincen_id + doc_type + page_number + bbox_norm in your front-end "
        "to jump/overlay highlights."
    )


# ---------------------------------- CLI ---------------------------------------

def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parse FinCEN PDFs (advisories, alerts, notices) into unified "
            "semantic + regex artifacts."
        )
    )
    p.add_argument(
        "pdf_dir",
        nargs="?",
        default=PDF_FOLDER,
        help="Primary folder containing PDFs (default: fincen_advisory_pdfs)",
    )
    p.add_argument(
        "out_csv",
        nargs="?",
        default=OUT_CSV,
        help="Summary CSV path (default: fincen_fraud_mapping.csv)",
    )
    p.add_argument(
        "out_json",
        nargs="?",
        default=OUT_JSON,
        help="Details JSON path (default: fincen_fraud_mapping_details.json)",
    )
    p.add_argument(
        "out_keywords",
        nargs="?",
        default=OUT_KEYWORD_LOCATIONS,
        help="Keyword locations CSV path",
    )
    p.add_argument(
        "out_semantic",
        nargs="?",
        default=OUT_SEMANTIC_CHUNKS,
        help="Semantic chunks CSV path",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all PDFs even if unchanged (ignore processing_state.json)",
    )
    p.add_argument(
        "--extra-pdf-dir",
        action="append",
        default=[],
        help=(
            "Additional folder of PDFs to include (can be repeated). "
            "Use this to pull in alerts/notices without reorganizing your files."
        ),
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
