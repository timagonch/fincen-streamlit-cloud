# src/extract_text.py
from pathlib import Path
from pdfminer.high_level import extract_text
import logging, re

# quiet pdfminer’s noisy messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def _tidy_preserve_newlines(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)   # collapse excessive blanks
    return s.strip()

def pdf_to_text(pdf_path: str) -> str:
    """Extract text from a single PDF, preserving newlines."""
    try:
        raw = extract_text(pdf_path) or ""
        return _tidy_preserve_newlines(raw)
    except Exception as e:
        logging.warning(f"Could not extract text from {pdf_path}: {e}")
        return ""

def batch_extract(in_dir="data/raw_pdfs", out_dir="data/extracted_text", max_chars=20000, clean_out=True):
    """
    Convert all PDFs in `in_dir` to .txt files in `out_dir`.
    Returns a list of dict rows: {pdf_path, txt_path, text}.
    """
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # avoid accumulating old runs
    if clean_out:
        for p in out_path.glob("*.txt"):
            try: p.unlink()
            except Exception: pass

    rows = []
    for pdf in in_path.glob("*.pdf"):
        text = pdf_to_text(str(pdf))[:max_chars]
        txt_path = out_path / (pdf.stem + ".txt")
        txt_path.write_text(text, encoding="utf-8")
        rows.append({"pdf_path": str(pdf), "txt_path": str(txt_path), "text": text})
    return rows
