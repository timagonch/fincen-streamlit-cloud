# src/fraud_detector.py
from __future__ import annotations
import json, hashlib, re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# keep it small/sane for this assignment
STOPWORDS = [
    "the","a","an","and","or","of","to","for","in","on","by","with","as","at","be","is","are","was","were",
    "from","that","this","it","its","their","they","them","he","she","we","you","your","our","not","no",
    "may","can","will","should","must","such","these","those","into","over","under","between","within",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _read_docs(txt_dir: str) -> List[tuple[str, str]]:
    pairs: List[tuple[str,str]] = []
    for p in sorted(Path(txt_dir).glob("*.txt")):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if t.strip():
                pairs.append((p.name, t))
        except Exception:
            pass  # skip unreadable
    return pairs

def detect(txt_dir: str, top_k: int = 8, min_df: int = 2) -> pd.DataFrame:
    """
    Learn common unigrams+bigrams and, per doc,
    surface top TF-IDF terms + the sentences that contain them.
    """
    docs = _read_docs(txt_dir)
    if not docs:
        return pd.DataFrame(columns=[
            "doc_id","file","text_chars","top_keywords",
            "flag_sentences","indicators_count","has_indicators"
        ])

    filenames, texts = zip(*docs)

    vec = TfidfVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"  # >=3 letters, allow hyphen
    )
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    rows: List[Dict[str, Any]] = []
    for i, (fname, text) in enumerate(docs):
        row = X[i].toarray().ravel()
        if row.sum() == 0:
            top_terms: List[str] = []
        else:
            top_idx = row.argsort()[::-1][:top_k]
            top_terms = [vocab[j] for j in top_idx if row[j] > 0]

        # simple sentence scan for any of the top terms
        sents = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 40]
        flags: List[str] = []
        if top_terms:
            term_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in top_terms) + r")\b", re.IGNORECASE)
            for s in sents:
                if term_re.search(s):
                    flags.append(s)

        rows.append({
            "doc_id": hashlib.md5(fname.encode("utf-8")).hexdigest()[:12],
            "file": fname,
            "text_chars": len(text),
            "top_keywords": json.dumps(top_terms),
            "flag_sentences": json.dumps(flags),
            "indicators_count": len(flags),
            "has_indicators": bool(flags),
        })

    return pd.DataFrame(rows)

