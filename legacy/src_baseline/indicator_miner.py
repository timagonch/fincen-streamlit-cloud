# src/indicator_miner.py
import re
from collections import Counter
from typing import List, Dict

STOP = set("""
a an the of to from for with by and or if on in into over under between during as is are was were be been being at this that these those their its it them
account accounts bank banks wire wires crypto funds cash card cards payment payments transfer transfers
""".split())

def _tok(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [w for w in s.split() if w not in STOP and len(w) > 2]

def _ngrams(tokens: List[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i+n])

def top_phrases_from_indicators(indicators: List[str], topk=10) -> List[str]:
    uni, bi, tri = Counter(), Counter(), Counter()
    for b in indicators:
        t = _tok(b)
        for x in t: uni[x] += 1
        for x in _ngrams(t, 2): bi[x] += 1
        for x in _ngrams(t, 3): tri[x] += 1

    out: List[str] = []

    def add(counter):
        for phrase, _ in counter.most_common():
            if not any(phrase in p or p in phrase for p in out):
                out.append(phrase)
            if len(out) >= topk: break

    add(tri)
    if len(out) < topk: add(bi)
    if len(out) < topk: add(uni)
    return out[:topk]

def flatten_sections(sections: Dict[str, List[str]]) -> List[str]:
    seen, out = set(), []
    for lst in sections.values():
        for s in lst:
            t = re.sub(r"\s+", " ", s).strip()
            k = t.lower()
            if t and k not in seen:
                out.append(t); seen.add(k)
    return out
