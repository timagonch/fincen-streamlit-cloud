# pip install requests beautifulsoup4
import os
import re
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag

LIST_URL = "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets"
UA = "Mozilla/5.0 (compatible; FinCEN-PDF-Grabber/1.4)"

def get(url, stream=False):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30, stream=stream, allow_redirects=True)
    r.raise_for_status()
    return r

def soup(url):
    return BeautifulSoup(get(url).text, "html.parser")

def looks_like_pdf(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".pdf")

def is_pdf_response(resp) -> bool:
    return "application/pdf" in resp.headers.get("Content-Type", "").lower()

def iter_non_spanish_fin_links(list_url):
    """Yield advisory links whose anchor text starts with FIN- and isn't Spanish."""
    s = soup(list_url)
    for a in s.select("a[href]"):
        text = a.get_text(strip=True)
        if text.startswith("FIN-") and "spanish" not in text.lower():
            yield urljoin(list_url, a["href"]), text

def find_first_pdf_on_page(page_url):
    """Prefer anchors whose text suggests English/non-Spanish; otherwise probe."""
    s = soup(page_url)
    # 1) Prefer clear non-Spanish PDF anchors
    for a in s.select("a[href]"):
        text = a.get_text(strip=True)
        href = urljoin(page_url, a["href"])
        if looks_like_pdf(href) and "spanish" not in text.lower():
            return href
    # 2) Any PDF anchors
    for a in s.select("a[href]"):
        href = urljoin(page_url, a["href"])
        if looks_like_pdf(href):
            return href
    # 3) Fallback by content-type probing a few links
    for a in s.select("a[href]")[:20]:
        href = urljoin(page_url, a["href"])
        try:
            r = get(href, stream=True)
            ok = is_pdf_response(r)
            r.close()
            if ok:
                return href
        except requests.RequestException:
            pass
    return None

def filename_from_response(resp, fallback_url):
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^"]+)"?', cd, flags=re.I)
    if m:
        return m.group(1)
    return os.path.basename(fallback_url.split("?", 1)[0]) or "fincen.pdf"

def download_pdf(pdf_url, out_dir="."):
    with get(pdf_url, stream=True) as r:
        if not is_pdf_response(r):
            raise RuntimeError(f"URL is not a PDF (Content-Type={r.headers.get('Content-Type')}).")
        os.makedirs(out_dir, exist_ok=True)
        fname = filename_from_response(r, pdf_url)
        # Avoid collisions by appending a counter if needed
        path = os.path.join(out_dir, fname)
        base, ext = os.path.splitext(path)
        i = 2
        while os.path.exists(path):
            path = f"{base} ({i}){ext}"
            i += 1
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return os.path.abspath(path)

def normalize_url(u: str) -> str:
    # Drop URL fragments and normalize query-insensitive duplicates
    return urldefrag(u)[0]

def main():
    parser = argparse.ArgumentParser(description="Download FinCEN advisory PDFs (non-Spanish, FIN- prefixed).")
    parser.add_argument("--max", type=int, default=1, help="Maximum number of PDFs to download (default: 1)")
    parser.add_argument("--out-dir", default="downloads", help="Output directory (default: downloads)")
    args = parser.parse_args()

    wanted = max(0, args.max)
    if wanted == 0:
        print("Nothing to do (--max 0).")
        return

    print("Loading listing page…")
    count = 0
    seen_pdf_urls = set()

    for link, text in iter_non_spanish_fin_links(LIST_URL):
        print(f"\nFound FIN link: {text} -> {link}")

        # Determine the PDF URL (direct or via advisory page)
        if looks_like_pdf(link):
            pdf_url = link
        else:
            print("Opening linked advisory page to find non-Spanish PDF…")
            pdf_url = find_first_pdf_on_page(link)

        if not pdf_url:
            print("  ⚠️  No PDF found on this advisory page, skipping.")
            continue

        norm_pdf = normalize_url(pdf_url)
        if norm_pdf in seen_pdf_urls:
            print("  ↪ Skipping duplicate PDF:", pdf_url)
            continue

        print(f"Downloading: {pdf_url}")
        try:
            saved = download_pdf(pdf_url, out_dir=args.out_dir)
            print(f"Saved to: {saved}")
            seen_pdf_urls.add(norm_pdf)
            count += 1
        except Exception as e:
            print(f"  ⚠️  Failed to download {pdf_url}: {e}")

        if count >= wanted:
            break

    if count == 0:
        print("No suitable PDFs were downloaded.")
    else:
        print(f"\nDone. Downloaded {count} PDF(s).")


def scrape(max_files: int = 10, out_dir: str = "data/raw_pdfs") -> int:
    """
    Download up to `max_files` eligible FinCEN advisory PDFs into `out_dir`.
    Returns the number of files downloaded. Uses the existing helper functions
    from this module (looks_like_pdf, find_first_pdf_on_page, download_pdf, etc.).
    """
    from pathlib import Path
    import time

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0

    for link, text in iter_non_spanish_fin_links(LIST_URL):
        # find a PDF url (direct or via the advisory page)
        if looks_like_pdf(link):
            pdf_url = link
        else:
            pdf_url = find_first_pdf_on_page(link)
        if not pdf_url:
            continue

        norm = normalize_url(pdf_url)
        if norm in seen:
            continue

        try:
            saved = download_pdf(pdf_url, out_dir=out_dir)
            seen.add(norm)
            count += 1
        except Exception as e:
            print("download failed:", e)

        if count >= max_files:
            break

        time.sleep(0.7)  # be polite

    return count


if __name__ == "__main__":
    # Allow running without CLI args in notebooks by providing defaults
    if hasattr(sys, 'argv') and len(sys.argv) == 1:
        sys.argv += ["--max", "1"]
    main()
