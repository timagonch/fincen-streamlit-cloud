import os
import csv
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # reads .env file

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")
TABLE_NAME = os.getenv("SUPABASE_TABLE")

LOCAL_PDF_DIR = "fincen_advisory_pdfs"
CSV_FILE = "fincen_advisories.csv"

# Optional: if your CSV date is in MM/DD/YYYY, convert to YYYY-MM-DD
DATE_INPUT_FORMAT = "%m/%d/%Y"
DATE_OUTPUT_FORMAT = "%Y-%m-%d"

# Prefix folder inside the bucket
BUCKET_PREFIX = "fincen_advisories"  # will store as: fincen_advisories/<filename>


# ---------- HELPERS ---------- #

def parse_date(date_str: str) -> str | None:
    """Convert date string from CSV to a DB-friendly format (YYYY-MM-DD)."""
    date_str = (date_str or "").strip()
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, DATE_INPUT_FORMAT)
        return dt.strftime(DATE_OUTPUT_FORMAT)
    except ValueError:
        # If parsing fails, just return the original string or None
        return date_str  # or return None


def upload_pdf_if_needed(supabase: Client, local_path: str, storage_path: str):
    """
    Upload a local PDF to Supabase Storage if not already there.
    We check by trying a download. If it errors, we upload.
    """
    # Normalize storage path (no leading slash)
    storage_path = storage_path.lstrip("/")

    # Check if exists by trying to download
    try:
        _ = supabase.storage.from_(BUCKET_NAME).download(storage_path)
        print(f"  [SKIP] Already in bucket: {storage_path}")
        return
    except Exception:
        # Not found or some error – we'll attempt upload
        pass

    print(f"  [UPLOAD] {storage_path}")
    with open(local_path, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "application/pdf"},
        )


def upsert_metadata_row(supabase: Client, row: dict):
    """
    Upsert metadata into the Supabase table by unique pdf_filename (or title).
    Adjust the 'on_conflict' column to what you consider unique.
    """
    on_conflict_col = "pdf_filename"  # make sure this column exists in TABLE_NAME

    print(f"  [UPSERT] {row['pdf_filename']} ({row['title']})")
    supabase.table(TABLE_NAME).upsert(row, on_conflict=on_conflict_col).execute()


# ---------- MAIN ---------- #

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set in .env")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    if not os.path.isdir(LOCAL_PDF_DIR):
        raise NotADirectoryError(f"PDF directory not found: {LOCAL_PDF_DIR}")

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            date_raw = row.get("date", "").strip()
            pdf_filename = row.get("pdf_filename", "").strip()
            pdf_url = row.get("pdf_url", "").strip()

            if not pdf_filename:
                print(f"[WARN] Missing pdf_filename for row: {title}")
                continue

            local_pdf_path = os.path.join(LOCAL_PDF_DIR, pdf_filename)
            if not os.path.exists(local_pdf_path):
                print(f"[WARN] PDF file not found locally: {local_pdf_path}")
                continue

            # Convert date if possible
            db_date = parse_date(date_raw)

            # Where we store it in Supabase Storage
            storage_path = f"{BUCKET_PREFIX}/{pdf_filename}"

            # 1) Upload the PDF file (if necessary)
            upload_pdf_if_needed(supabase, local_pdf_path, storage_path)

            # 2) Upsert the metadata row into the table
            metadata_row = {
                "title": title,
                "date": db_date,
                "pdf_filename": pdf_filename,
                "pdf_url": pdf_url,
                "storage_path": storage_path,
            }
            upsert_metadata_row(supabase, metadata_row)


if __name__ == "__main__":
    main()
