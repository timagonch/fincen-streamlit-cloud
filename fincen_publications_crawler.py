#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fincen_publications_crawler.py
--------------------------------

Unified crawler for FinCEN Advisories + Alerts + Notices.

Logic (aligned with your old crawlers):

- Advisories:
    * Crawl listing pages (reverse-chronological table).
    * Skip Spanish + withdrawals.
    * Early-stop based on Supabase (title_norm + date).
    * For each NEW advisory: follow detail page, locate PDF, download
      (or reuse local), upload to Supabase bucket, upsert into
      `fincen_publications`.

- Alerts + Notices:
    * Parse https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets
      like the old alerts_notices_crawler.
    * Identify FIN-YYYY-AlertNN / FIN-YYYY-NTC / FIN-YYYY-CTA.
    * For each NEW fincen_id (not in Supabase): follow detail page,
      safely locate PDF (using same pattern as old code but more robust),
      download / upload, upsert.

NO local CSVs. Supabase is the source of truth.

Env:
    SUPABASE_URL
    SUPABASE_KEY  (or SUPABASE_ANON_KEY for testing)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from supabase_helpers import (
    get_supabase_client,
    normalize_title_for_key,
    load_existing_advisory_keys_from_supabase,
    load_existing_ids_from_supabase,
    upsert_publication,
    upload_pdf_to_bucket,
    compute_sha256,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.fincen.gov"

USER_AGENT = (
    "FinCEN-Fraud-Library/1.0 "
    "(academic; UNC Charlotte DSBA; contact team)"
)

ADVISORY_DIR = "fincen_advisory_pdfs"
ALERT_DIR = "fincen_alert_pdfs"
NOTICE_DIR = "fincen_notice_pdfs"

ADVISORY_START_URL = (
    "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets/"
    "advisories?field_date_release_value=&field_date_release_value_1="
    "&field_tags_advisory_target_id=All&page=0"
)

REQUEST_DELAY_PAGE = 0.5
REQUEST_DELAY_PDF = 0.4
REQUEST_DELAY_DETAIL = 0.3


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Thin wrapper, like in old crawlers."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def sanitize_filename(name: str) -> str:
    clean = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "-", "_", "."):
            clean.append(ch)
        else:
            clean.append("_")
    return "_".join("".join(clean).split())[:180]


def filename_from_pdf_url(pdf_url: str, title: Optional[str] = None) -> str:
    path = urlparse(pdf_url).path
    base = os.path.basename(path)
    if base.lower().endswith(".pdf"):
        return base
    if title:
        return sanitize_filename(title) + ".pdf"
    return "fincen.pdf"


def download_pdf(session: requests.Session, pdf_url: str, dest: Path) -> None:
    """
    Download a single PDF if it doesn't already exist.

    We keep this behavior from the old crawler: reuse local file if present,
    but we still compute sha256 + upload + upsert afterwards.
    """
    if dest.exists():
        print(f"      [USE LOCAL] {dest.name} (already downloaded)")
        return

    print(f"      [GET] {pdf_url}")
    r = session.get(pdf_url, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    time.sleep(REQUEST_DELAY_PDF)


# ---------------------------------------------------------------------------
# Advisory helpers (mirroring old advisory_crawler logic)
# ---------------------------------------------------------------------------

def find_advisory_table(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue
        if "title" in headers[0] and "date" in "".join(headers):
            return table
    return None


def extract_advisory_rows(table: BeautifulSoup, page_url: str):
    body = table.find("tbody") or table
    for row in body.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        a = cells[0].find("a", href=True)
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        detail_url = urljoin(page_url, a["href"])
        date = cells[1].get_text(" ", strip=True)
        yield title, detail_url, date


def find_next_advisory_page(soup: BeautifulSoup, current_url: str) -> Optional[str]:
    pager = soup.find("ul", class_="pager")
    if not pager:
        return None
    nxt = pager.find("li", class_="pager-next")
    if not nxt:
        return None
    a = nxt.find("a", href=True)
    if not a:
        return None
    return urljoin(current_url, a["href"])


FINCEN_ID_PATTERN = re.compile(r"FIN-\d{4}-\w+", re.IGNORECASE)


def extract_fincen_id_from_title(title: str) -> str:
    m = FINCEN_ID_PATTERN.search(title or "")
    return m.group(0).upper() if m else ""


def find_pdf_url_for_advisory(session: requests.Session, detail_url: str) -> Optional[str]:
    """
    Old advisory_crawler logic: fetch HTML detail page and look for PDF links.
    """
    try:
        soup = fetch_soup(session, detail_url)
    except Exception as e:
        print(f"      [WARN] Failed to fetch advisory detail {detail_url}: {e}")
        return None

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.lower().endswith(".pdf")
            or "/system/files" in href
            or "/sites/default/files" in href
        ):
            return urljoin(detail_url, href)

    return None


# ---------------------------------------------------------------------------
# Crawl Advisories (Supabase early-stop)
# ---------------------------------------------------------------------------

def crawl_advisories(session: requests.Session, client) -> None:
    print("\n=== Crawling Advisories ===")

    existing_keys = load_existing_advisory_keys_from_supabase(client)
    print(f"[INFO] Advisory keys loaded from Supabase: {len(existing_keys)}")

    # --------------------------------------------
    # Step 1: Look only at page=0 to check newest advisory
    # --------------------------------------------
    page0_url = ADVISORY_START_URL
    print(f"\n[Check Page 0 for early stop] {page0_url}")

    try:
        soup = fetch_soup(session, page0_url)
    except Exception as e:
        print(f"[WARN] Failed to load advisory page 0: {e}")
        return

    table = find_advisory_table(soup)
    if not table:
        print("[WARN] No advisory table found on page 0.")
        return

    rows = list(extract_advisory_rows(table, page0_url))
    if not rows:
        print("[WARN] No advisory rows on page 0.")
        return

    # Get only the first (newest) advisory from page 0
    newest_title, newest_detail, newest_date = rows[0]
    newest_norm = normalize_title_for_key(newest_title)
    newest_date_str = (newest_date or "").strip()

    print(f"[Newest Advisory] {newest_title} ({newest_date_str})")

    # --------------------------------------------
    # EARLY STOP CONDITION
    # --------------------------------------------
    if (newest_norm, newest_date_str) in existing_keys:
        print("[✓] Advisory list already up-to-date. Skipping all advisory crawling.")
        return

    # --------------------------------------------
    # If not found → full crawl (like your updated version)
    # --------------------------------------------
    print("[INFO] New advisory detected — full crawl needed.")

    seen_pdf_urls: Set[str] = set()
    max_pages = 20

    for page_idx in range(max_pages):
        url = ADVISORY_START_URL.replace("page=0", f"page={page_idx}")
        print(f"\n[Page {page_idx}] {url}")

        try:
            soup = fetch_soup(session, url)
        except Exception as e:
            print(f"  [WARN] Failed to fetch page {url}: {e}")
            break

        table = find_advisory_table(soup)
        if not table:
            print("  [INFO] No advisory table — stopping.")
            break

        page_rows = list(extract_advisory_rows(table, url))
        if not page_rows:
            print("  [INFO] No rows — stopping.")
            break

        for title, detail_url, date in page_rows:
            # normalize
            title_norm = normalize_title_for_key(title)
            date_str = (date or "").strip()

            # skip Spanish / Withdrawal
            if "spanish" in title_norm:
                continue
            if "withdrawal" in title_norm:
                continue

            # skip if already in Supabase
            if (title_norm, date_str) in existing_keys:
                print(f"  [SKIP existing advisory] {title} ({date_str})")
                continue

            # NEW advisory → process normally
            print(f"  [NEW] {title} ({date_str})")

            pdf_url = find_pdf_url_for_advisory(session, detail_url)
            if not pdf_url:
                print("       [WARN] No PDF found.")
                continue

            if pdf_url in seen_pdf_urls:
                print("       [SKIP] Duplicate PDF URL.")
                continue
            seen_pdf_urls.add(pdf_url)

            pdf_filename = filename_from_pdf_url(pdf_url, title)
            pdf_path = Path(ADVISORY_DIR) / pdf_filename

            fincen_id = extract_fincen_id_from_title(title)

            download_pdf(session, pdf_url, pdf_path)
            sha256 = compute_sha256(pdf_path)

            remote_path = f"advisories/{pdf_filename}"
            upload_pdf_to_bucket(client, pdf_path, remote_path)

            upsert_publication(
                client,
                fincen_id=fincen_id,
                title=title,
                date=date_str,
                doc_type="Advisory",
                pdf_filename=pdf_filename,
                pdf_url=pdf_url,
                detail_url=detail_url,
                sha256=sha256,
            )

            existing_keys.add((title_norm, date_str))

        time.sleep(REQUEST_DELAY_PAGE)

    print("[✓] Advisory crawl complete.")




# ---------------------------------------------------------------------------
# Alerts & Notices helpers (based on old alerts_notices_crawler logic)
# ---------------------------------------------------------------------------

LIST_URL = BASE_URL + "/resources/advisoriesbulletinsfact-sheets"


def _parse_row_for_entry(a_tag, fincen_id: str, doc_type: str) -> Optional[Dict[str, str]]:
    """
    Given the <a> tag that contains the FIN code, climb to the parent <tr>
    and parse the Date + Title cells (same approach as old code).
    """
    row = a_tag.find_parent("tr")
    if row is None:
        return None

    cells = row.find_all("td")
    if len(cells) < 3:
        return None

    date = cells[1].get_text(" ", strip=True)
    title = cells[2].get_text(" ", strip=True)
    detail_url = urljoin(BASE_URL, a_tag["href"])

    return {
        "fincen_id": fincen_id,
        "doc_type": doc_type,
        "title": title,
        "date": date,
        "detail_url": detail_url,
    }


def _extract_alert_entries(soup: BeautifulSoup) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"FIN-\d{4}-ALERT\d+", re.IGNORECASE)

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
    results: List[Dict[str, str]] = []
    pattern = re.compile(r"FIN-\d{4}-(NTC\d+|CTA\d+)", re.IGNORECASE)

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


def find_pdf_for_alert_notice(session: requests.Session, detail_url: str) -> Optional[str]:
    """
    Robust version of old find_pdf_url:

      - If the detail_url already points to a .pdf, return it.
      - If the response is 'application/pdf', return detail_url.
      - Otherwise, parse HTML (with old fetch_soup pattern) and
        look for a PDF link.
      - If parsing fails or nothing found, return None (no crash).
    """
    parsed = urlparse(detail_url)
    if parsed.path.lower().endswith(".pdf"):
        return detail_url

    try:
        resp = session.get(detail_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch detail page {detail_url}: {e}")
        return None

    content_type = resp.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type:
        # The "detail" page is literally a PDF
        return detail_url

    # Try to parse as HTML, like old code, but catch parser errors
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[WARN] Failed to parse HTML for {detail_url}: {e}")
        return None

    time.sleep(REQUEST_DELAY_DETAIL)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.lower().endswith(".pdf")
            or "/system/files" in href
            or "/sites/default/files" in href
        ):
            return urljoin(detail_url, href)

    return None


def extract_alerts_notices(session: requests.Session):
    """
    Hit the listing page and extract alert + notice entries using the same
    FIN-code + <tr> logic as the old alerts_notices_crawler.
    """
    soup = fetch_soup(session, LIST_URL)
    alerts = _extract_alert_entries(soup)
    notices = _extract_notice_entries(soup)

    # Deduplicate by fincen_id in case the page shows them twice
    alerts_dedup: Dict[str, Dict[str, str]] = {}
    for a in alerts:
        alerts_dedup[a["fincen_id"]] = a

    notices_dedup: Dict[str, Dict[str, str]] = {}
    for n in notices:
        notices_dedup[n["fincen_id"]] = n

    return list(alerts_dedup.values()), list(notices_dedup.values())


# ---------------------------------------------------------------------------
# Crawl Alerts & Notices (Supabase-driven newness)
# ---------------------------------------------------------------------------

def crawl_alerts_and_notices(session: requests.Session, client) -> None:
    print("\n=== Crawling Alerts & Notices ===")

    alerts, notices = extract_alerts_notices(session)
    print(f"[INFO] Listing shows {len(alerts)} alerts, {len(notices)} notices")

    existing_alert_ids = load_existing_ids_from_supabase(client, "Alert")
    existing_notice_ids = load_existing_ids_from_supabase(client, "Notice")

    # ----- ALERTS -----
    for item in alerts:
        fid = item["fincen_id"]
        if fid in existing_alert_ids:
            print(f"[SKIP Alert] Already in Supabase: {fid}")
            continue

        print(f"\n[NEW Alert] {fid}  {item['title']} ({item['date']})")
        pdf_url = find_pdf_for_alert_notice(session, item["detail_url"])
        if not pdf_url:
            print(f"    [WARN] No PDF found for {fid}")
            continue

        pdf_filename = filename_from_pdf_url(pdf_url, item["title"])
        pdf_path = Path(ALERT_DIR) / pdf_filename

        download_pdf(session, pdf_url, pdf_path)
        sha256 = compute_sha256(pdf_path)

        remote_path = f"alerts/{pdf_filename}"
        try:
            upload_pdf_to_bucket(client, pdf_path, remote_path)
        except Exception as e:
            print(f"    [WARN] Failed to upload alert PDF: {e}")

        upsert_publication(
            client,
            fincen_id=fid,
            title=item["title"],
            date=item["date"],
            doc_type="Alert",
            pdf_filename=pdf_filename,
            pdf_url=pdf_url,
            detail_url=item["detail_url"],
            sha256=sha256,
        )

        existing_alert_ids.add(fid)

    # ----- NOTICES -----
    for item in notices:
        fid = item["fincen_id"]
        if fid in existing_notice_ids:
            print(f"[SKIP Notice] Already in Supabase: {fid}")
            continue

        print(f"\n[NEW Notice] {fid}  {item['title']} ({item['date']})")
        pdf_url = find_pdf_for_alert_notice(session, item["detail_url"])
        if not pdf_url:
            print(f"    [WARN] No PDF found for {fid}")
            continue

        pdf_filename = filename_from_pdf_url(pdf_url, item["title"])
        pdf_path = Path(NOTICE_DIR) / pdf_filename

        download_pdf(session, pdf_url, pdf_path)
        sha256 = compute_sha256(pdf_path)

        remote_path = f"notices/{pdf_filename}"
        try:
            upload_pdf_to_bucket(client, pdf_path, remote_path)
        except Exception as e:
            print(f"    [WARN] Failed to upload notice PDF: {e}")

        upsert_publication(
            client,
            fincen_id=fid,
            title=item["title"],
            date=item["date"],
            doc_type="Notice",
            pdf_filename=pdf_filename,
            pdf_url=pdf_url,
            detail_url=item["detail_url"],
            sha256=sha256,
        )

        existing_notice_ids.add(fid)

    print("\n[✓] Alerts & Notices crawl complete.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("==== FinCEN Unified Crawler ====")
    client = get_supabase_client()
    session = make_session()

    crawl_advisories(session, client)
    crawl_alerts_and_notices(session, client)

    print("\n[✓] Crawler finished. Supabase is up-to-date.")


if __name__ == "__main__":
    main()
