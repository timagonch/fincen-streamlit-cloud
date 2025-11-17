#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
advisory_crawler.py

Crawler for FinCEN *Advisories* that populates a canonical
`fincen_advisories.csv` compatible with the unified publications schema:

    fincen_id, title, date, pdf_filename, pdf_url, publication_type

Design choices:
  * Primary join/dedup key for analytics is pdf_filename/pdf_url,
    because older advisories (Issue 26, etc.) don't have FIN- codes.
  * fincen_id is best-effort metadata, extracted when present (FIN-YYYY-XXX).
  * We skip Spanish-language duplicates (title containing "spanish").
  * We always rewrite the CSV on each run (idempotent).
  * To keep re-runs fast, we reuse existing CSV rows and avoid re-scraping
    advisory pages we've already seen.
"""

from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------- CONFIG ---------------- #

# First page of the Advisories listing (filters "Any" advisory type).
START_URL = (
    "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets/"
    "advisories?field_date_release_value=&field_date_release_value_1="
    "&field_tags_advisory_target_id=All&page=0"
)

DOWNLOAD_DIR = "fincen_advisory_pdfs"
CSV_FILE = "fincen_advisories.csv"

REQUEST_DELAY_PAGE = 0.5   # delay between listing pages
REQUEST_DELAY_ITEM = 0.0   # delay between individual advisories

PUBLICATION_TYPE = "advisory"

# ---------------- HTTP SESSION ---------------- #

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FinCEN-AdvisoryCrawler/1.0; "
            "+https://github.com/your-org)"
        )
    }
)

# ---------------- HELPERS ---------------- #


def get_soup(url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_advisory_table(soup: BeautifulSoup):
    """
    Locate the main Title/Date/Description table on the page.
    """
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if {"Title", "Date", "Description"}.issubset(headers):
            return table
    return None


def extract_rows_from_table(table):
    tbody = table.find("tbody") or table
    return tbody.find_all("tr")


def extract_row_data(row, base_url: str):
    """
    Extract (title, advisory_url, date) from a table row.
    """
    cells = row.find_all("td")
    if not cells or len(cells) < 2:
        return None

    a = cells[0].find("a", href=True)
    if not a:
        return None

    title = a.get_text(strip=True)
    advisory_url = urljoin(base_url, a["href"])
    date = cells[1].get_text(strip=True)

    return title, advisory_url, date


def find_pdf_url(advisory_url: str) -> str | None:
    """
    Open the advisory detail page and find the first PDF link.
    """
    soup = get_soup(advisory_url)

    # strict: href endswith .pdf
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(advisory_url, href)

    # fallback: .pdf appears anywhere
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if ".pdf" in href:
            return urljoin(advisory_url, a["href"])

    return None


def sanitize_filename(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return "_".join(name.split())[:180]


def filename_from_pdf_url(pdf_url: str, title: str | None = None) -> str:
    """
    Prefer the last path segment of the PDF URL; otherwise sanitize the title.
    """
    path = urlparse(pdf_url).path
    last_part = os.path.basename(path)
    if last_part and last_part.lower().endswith(".pdf"):
        return last_part
    if title:
        return sanitize_filename(title) + ".pdf"
    return "advisory.pdf"


def download_pdf(pdf_url: str, filepath: str) -> None:
    """
    Download a single PDF if it doesn't already exist.
    """
    if os.path.exists(filepath):
        return

    print(f"  [PDF ] Downloading {os.path.basename(filepath)}")
    resp = session.get(pdf_url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)


# ---------------- ID & PAGINATION HELPERS ---------------- #

FINCEN_ID_PAT = re.compile(r"FIN-\d{4}-[A-Za-z0-9]+", re.IGNORECASE)


def extract_fincen_id(text: str) -> str:
    """
    Extract FIN-YYYY-XXX style identifier if present in a title.
    Older advisories (Issue 26, etc.) won't have one; we return "".
    """
    m = FINCEN_ID_PAT.search(text or "")
    return m.group(0).upper() if m else ""


def find_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """
    Find the "Next" page link in the advisory listing.
    """
    a = soup.find("a", rel="next")
    if a and a.get("href"):
        return urljoin(current_url, a["href"])

    # Fallback: any link whose text contains "next"
    for a in soup.find_all("a", href=True):
        if "next" in a.get_text(strip=True).lower():
            return urljoin(current_url, a["href"])
    return None


# ---------------- EXISTING CSV REUSE (SPEED-UP) ---------------- #


def load_existing_rows(path: str = CSV_FILE) -> dict[str, dict]:
    """
    Load existing advisory rows from the canonical CSV, keyed by
    normalized title (lowercased, stripped).

    This lets us avoid re-scraping advisory pages we already know about,
    while still rewriting the CSV each run.
    """
    p = Path(path)
    if not p.exists():
        return {}

    rows: dict[str, dict] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip().lower()
            if not title:
                continue
            rows[title] = row
    return rows


# ---------------- MAIN CRAWLER ---------------- #


def crawl_all_advisory_pdfs(start_url: str = START_URL) -> None:
    """
    Walk the paginated advisories listing, download PDFs, and
    write fincen_advisories.csv in the unified schema.

    CSV columns:
        fincen_id, title, date, pdf_filename, pdf_url, publication_type
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Download folder: {os.path.abspath(DOWNLOAD_DIR)}")

    existing_rows = load_existing_rows(CSV_FILE)
    print(f"Loaded {len(existing_rows)} existing advisory rows from CSV.")

    visited_pages: set[str] = set()
    seen_pdf_urls: set[str] = set()
    records: list[dict] = []

    url = start_url
    page_idx = 0

    while url and url not in visited_pages:
        page_idx += 1
        print(f"\n[PAGE {page_idx}] {url}")
        visited_pages.add(url)

        soup = get_soup(url)
        table = find_advisory_table(soup)
        if not table:
            print("  [WARN] Could not find advisory table on this page.")
            break

        rows = extract_rows_from_table(table)
        print(f"  Found {len(rows)} rows on this page.")

        for row in rows:
            parsed = extract_row_data(row, url)
            if not parsed:
                continue

            title, advisory_url, date = parsed
            title_stripped = (title or "").strip()
            title_norm = title_stripped.lower()

            # Skip Spanish-language duplicates
            if "spanish" in title_norm:
                print(f"  [SKIP Spanish] {title_stripped}")
                continue
            # Skip Advisory Withdrawal notices
            if "advisory withdrawal" in title_norm:
                print(f"  [SKIP Withdrawal] {title_stripped}")
                continue

            # If we have this title in an existing CSV, reuse it and don't re-scrape
            if title_norm in existing_rows:
                prev = existing_rows[title_norm]
                records.append(
                    {
                        "fincen_id": prev.get("fincen_id", ""),
                        "title": prev.get("title", title_stripped),
                        "date": prev.get("date", date),
                        "pdf_filename": prev.get("pdf_filename", ""),
                        "pdf_url": prev.get("pdf_url", ""),
                        "publication_type": prev.get(
                            "publication_type", PUBLICATION_TYPE
                        ),
                    }
                )
                print(f"  [SKIP existing] {title_stripped}")
                continue

            print(f"  [ITEM] {title_stripped} ({date})")
            print(f"        Advisory page: {advisory_url}")

            # New advisory -> we must scrape its detail page to find the PDF
            try:
                pdf_url = find_pdf_url(advisory_url)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERR ] Failed to parse advisory page {advisory_url}: {e}")
                continue

            if not pdf_url:
                print(f"  [WARN] No PDF link found on advisory page: {advisory_url}")
                continue

            if pdf_url in seen_pdf_urls:
                print(f"  [SKIP dup pdf_url] {title_stripped}")
                continue
            seen_pdf_urls.add(pdf_url)

            pdf_filename = filename_from_pdf_url(pdf_url, title_stripped)
            filepath = os.path.join(DOWNLOAD_DIR, pdf_filename)

            fincen_id = extract_fincen_id(title_stripped)

            print(f"        fincen_id={fincen_id or '—'}")
            print(f"        pdf={pdf_filename}")

            try:
                download_pdf(pdf_url, filepath)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERR ] Failed to download PDF {pdf_url}: {e}")
                continue

            records.append(
                {
                    "fincen_id": fincen_id,
                    "title": title_stripped,
                    "date": date,
                    "pdf_filename": pdf_filename,
                    "pdf_url": pdf_url,
                    "publication_type": PUBLICATION_TYPE,
                }
            )

            if REQUEST_DELAY_ITEM > 0:
                time.sleep(REQUEST_DELAY_ITEM)

        # Next page
        next_url = find_next_page_url(soup, url)
        if not next_url or next_url in visited_pages:
            print("\n[DONE] No more pages.")
            break

        url = next_url
        time.sleep(REQUEST_DELAY_PAGE)

    write_csv(records, CSV_FILE)


def write_csv(records: list[dict], path: str) -> None:
    """
    Write the canonical advisories CSV; overwrite any existing file.
    """
    fieldnames = [
        "fincen_id",
        "title",
        "date",
        "pdf_filename",
        "pdf_url",
        "publication_type",
    ]

    print(f"\n[WRITE] {path} ({len(records)} rows)")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    crawl_all_advisory_pdfs()
