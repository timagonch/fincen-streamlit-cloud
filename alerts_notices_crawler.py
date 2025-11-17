#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alerts_notices_crawler.py
---------------------------------
Scrape FinCEN **Alerts** and **FinCEN Notices** from:

    https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets

For each alert / notice this script will:

  1) Parse the listing page and identify the FIN-XXXX-Alert / FIN-XXXX-NTC / FIN-XXXX-CTA codes.
  2) Follow each link to the detail page (if not a direct PDF).
  3) Find the primary PDF link and download it to:
        - fincen_alert_pdfs/   (for alerts)
        - fincen_notice_pdfs/  (for notices)
  4) Append basic metadata to CSVs:
        - fincen_alerts.csv
        - fincen_notices.csv

CSV columns:

    fincen_id     e.g. "FIN-2025-Alert002" or "FIN-2024-NTC2"
    doc_type      "Alert" or "Notice"
    title         description text from the row (3rd column)
    date          release date string as shown on the site (e.g. "05/01/2025")
    pdf_filename  downloaded file name, e.g. "Alert-FinCEN-Scams-FINAL508.pdf"
    pdf_url       full URL to the PDF
    detail_url    URL of the HTML detail page (or the PDF itself if no HTML page)

Notes:
  * Spanish-language duplicates are skipped (rows whose FULL row text contains "Spanish").
  * The script is idempotent: re-running will skip items whose fincen_id is already in the CSV,
    and it will not re-download existing PDFs (it checks by filename).
"""

import csv
import os
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.fincen.gov"
LIST_URL = "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets"

ALERT_CSV = "fincen_alerts.csv"
NOTICE_CSV = "fincen_notices.csv"

ALERT_DIR = "fincen_alert_pdfs"
NOTICE_DIR = "fincen_notice_pdfs"

REQUEST_DELAY_DETAIL = 0.3   # delay between detail-page requests
REQUEST_DELAY_PDF = 0.5      # delay between PDF downloads


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "FinCEN-Fraud-Library/1.0 (+for academic research; contact your-team@example.com)"
        }
    )
    return s


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


# ---------------------------------------------------------------------------
# Row parser — this is the key fix: we go to the <tr> and read the Date cell
# ---------------------------------------------------------------------------

def _parse_row_for_entry(
    a_tag,
    fincen_id: str,
    doc_type: str,
) -> Optional[Dict[str, str]]:
    """
    Given the <a> tag that contains the FIN code, climb to the parent <tr>
    and pull:
      - date from 2nd <td> (index 1)
      - description from 3rd <td> (index 2) as the "title"
    """
    row = a_tag.find_parent("tr")
    if row is None:
        return None

    row_text = row.get_text(" ", strip=True).lower()
    if "spanish" in row_text:
        # Skip the Spanish-language row entirely
        return None

    cells = row.find_all("td")
    if len(cells) < 2:
        return None

    # Column layout on FinCEN:
    #   [0] Title (the FIN-YYYY-AlertXXX link)
    #   [1] Date (e.g. 05/01/2025)
    #   [2] Description (we'll use as our human-readable title)
    date = cells[1].get_text(strip=True)

    if len(cells) >= 3:
        title = cells[2].get_text(" ", strip=True)
    else:
        # Fallback: if no separate description column, use the Title cell text
        title = cells[0].get_text(" ", strip=True)

    detail_url = urljoin(BASE_URL, a_tag["href"])

    return {
        "fincen_id": fincen_id,
        "doc_type": doc_type,
        "title": title,
        "date": date,
        "detail_url": detail_url,
    }


# ---------------------------------------------------------------------------
# Extract entries from listing page
# ---------------------------------------------------------------------------

def _extract_alert_entries(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Find all alert rows on the listing page.

    We look for anchor tags whose text looks like FIN-YYYY-AlertXXX and
    then parse the row <tr> they live in.
    """
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"FIN-\d{4}-Alert\d+", re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        m = pattern.search(text)
        if not m:
            continue

        fincen_id = m.group(0).upper()
        entry = _parse_row_for_entry(a, fincen_id, "Alert")
        if entry:
            results.append(entry)

    return results


def _extract_notice_entries(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Find all FinCEN Notice rows on the listing page.

    We look for anchor tags that:
      * contain NTC or CTA codes like FIN-2024-NTC2 or FIN-2025-CTA1
      * then parse the <tr> they live in.
    """
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"FIN-\d{4}-(?:NTC|CTA)\d+", re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        m = pattern.search(text)
        if not m:
            continue

        fincen_id = m.group(0).upper()
        entry = _parse_row_for_entry(a, fincen_id, "Notice")
        if entry:
            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# PDF discovery + download
# ---------------------------------------------------------------------------

def find_pdf_url(session: requests.Session, detail_url: str) -> Optional[str]:
    """
    Given the detail URL (or possibly a direct PDF URL),
    return the resolved PDF URL or None if not found.
    """
    # If the link is already a PDF, just use it
    parsed = urlparse(detail_url)
    if parsed.path.lower().endswith(".pdf"):
        return detail_url

    # Otherwise, fetch the detail page and search for a PDF link
    try:
        soup = fetch_soup(session, detail_url)
        time.sleep(REQUEST_DELAY_DETAIL)
    except Exception as e:
        print(f"[WARN] Failed to fetch detail page {detail_url}: {e}")
        return None

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Many FinCEN PDFs live under /system/files or /sites/default/files
        if (
            href.lower().endswith(".pdf")
            or "/system/files" in href
            or "/sites/default/files" in href
        ):
            return urljoin(detail_url, href)

    return None


def download_pdf(
    session: requests.Session, pdf_url: str, dest_dir: str
) -> Optional[str]:
    """
    Download the PDF into dest_dir, returning the filename (not full path)
    or None if the download fails.
    """
    os.makedirs(dest_dir, exist_ok=True)

    parsed = urlparse(pdf_url)
    filename = os.path.basename(parsed.path)
    if not filename:
        filename = "fincen_doc.pdf"

    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        print(f"[SKIP] PDF already exists: {filename}")
        return filename

    try:
        with session.get(pdf_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] Downloaded: {filename}")
        time.sleep(REQUEST_DELAY_PDF)
        return filename
    except Exception as e:
        print(f"[ERROR] Failed to download {pdf_url}: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_existing_ids(csv_path: str) -> Set[str]:
    """
    Load existing fincen_id values from the given CSV (if it exists)
    to support idempotent runs.
    """
    ids: Set[str] = set()
    if not os.path.exists(csv_path):
        return ids

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("fincen_id") or "").strip()
            if fid:
                ids.add(fid)
    return ids


def append_row(csv_path: str, row: Dict[str, str]) -> None:
    """
    Append a single row to the CSV, writing the header if the file is new.
    """
    fieldnames = [
        "fincen_id",
        "doc_type",
        "title",
        "date",
        "pdf_filename",
        "pdf_url",
        "detail_url",
    ]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------

def crawl_alerts_and_notices() -> None:
    session = make_session()
    print(f"[INFO] Fetching listing page: {LIST_URL}")
    soup = fetch_soup(session, LIST_URL)

    alerts = _extract_alert_entries(soup)
    notices = _extract_notice_entries(soup)

    print(f"[INFO] Found {len(alerts)} alerts and {len(notices)} notices on listing page")

    existing_alert_ids = load_existing_ids(ALERT_CSV)
    existing_notice_ids = load_existing_ids(NOTICE_CSV)

    # ---- Alerts ----
    for item in alerts:
        fid = item["fincen_id"]
        if fid in existing_alert_ids:
            print(f"[SKIP] Alert already in CSV: {fid}")
            continue

        print(f"\n[ALERT] Processing {fid}")
        pdf_url = find_pdf_url(session, item["detail_url"])
        if not pdf_url:
            print(f"[WARN] No PDF found for {fid} ({item['detail_url']})")
            continue

        pdf_filename = download_pdf(session, pdf_url, ALERT_DIR)
        if not pdf_filename:
            continue

        row = {
            "fincen_id": fid,
            "doc_type": "Alert",
            "title": item["title"],
            "date": item["date"],
            "pdf_filename": pdf_filename,
            "pdf_url": pdf_url,
            "detail_url": item["detail_url"],
        }
        append_row(ALERT_CSV, row)
        existing_alert_ids.add(fid)

    # ---- Notices ----
    for item in notices:
        fid = item["fincen_id"]
        if fid in existing_notice_ids:
            print(f"[SKIP] Notice already in CSV: {fid}")
            continue

        print(f"\n[NOTICE] Processing {fid}")
        pdf_url = find_pdf_url(session, item["detail_url"])
        if not pdf_url:
            print(f"[WARN] No PDF found for {fid} ({item['detail_url']})")
            continue

        pdf_filename = download_pdf(session, pdf_url, NOTICE_DIR)
        if not pdf_filename:
            continue

        row = {
            "fincen_id": fid,
            "doc_type": "Notice",
            "title": item["title"],
            "date": item["date"],
            "pdf_filename": pdf_filename,
            "pdf_url": pdf_url,
            "detail_url": item["detail_url"],
        }
        append_row(NOTICE_CSV, row)
        existing_notice_ids.add(fid)


if __name__ == "__main__":
    crawl_alerts_and_notices()
