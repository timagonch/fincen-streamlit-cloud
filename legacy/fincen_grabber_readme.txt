# FinCEN Advisory PDF Downloader

Download one or more **English (non‑Spanish) FinCEN advisories/bulletins/fact‑sheet PDFs** from the official listing page.

This script:
- Crawls the FinCEN **Advisories, Bulletins & Fact Sheets** listing.
- Filters for items whose link text **begins with `FIN-`** and **does not contain “Spanish”**.
- Follows through to the advisory page (when needed) to locate the **first non‑Spanish PDF**.
- Verifies **Content‑Type** to ensure the file is actually a PDF.
- Skips duplicates, handles filename collisions (`name (2).pdf`, etc.), and continues on individual failures.
- Lets you control **how many PDFs** to download via `--max`.

---

## Requirements

- Python **3.8+** (tested with 3.10+)
- Packages:
  - `requests`
  - `beautifulsoup4`

Install them (in a virtual environment is recommended):

```bash
pip install requests beautifulsoup4
```

---

## Quick Start

Download the **latest single** eligible advisory PDF to `./downloads` (default behavior):

```bash
python fincen_grabber.py
```

Download **five** PDFs:

```bash
python fincen_grabber.py --max 5
```

Download **three** PDFs to a custom folder:

```bash
python fincen_grabber.py --max 3 --out-dir fincen_pdfs
```

> Tip: If you run the script with no CLI arguments inside certain notebook environments,
> it defaults to `--max 1` to mimic the original single‑file behavior.

---

## Command‑Line Options

| Option | Type | Default | Description |
|---|---:|---|---|
| `--max` | int | `1` | Maximum number of PDFs to download. Use `0` to perform no downloads (prints a message and exits). |
| `--out-dir` | str | `downloads` | Output directory where PDFs will be saved. Created if it does not exist. |

---

## How It Works (High‑Level)

1. **Load listing page**: `https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets`  
2. **Find eligible advisory links**: anchor text that **starts with `FIN-`** and does **not** include “Spanish”.  
3. For each advisory (until `--max` is reached):
   - If the link is a **direct PDF**, use it; otherwise, **open the advisory page** and search for the **first non‑Spanish PDF** link.
   - If necessary, probe up to 20 links by HTTP **Content‑Type** to confirm a real PDF.
   - **Normalize** the PDF URL (remove `#fragment`) and **skip duplicates**.
   - **Download** with streaming, verify `Content-Type: application/pdf`, and **write the file** to disk.
   - If a file with the same name already exists, **append a counter** (`(2)`, `(3)`, …).

The script uses a distinct **User‑Agent** string (`FinCEN-PDF-Grabber/1.4`) and reasonable timeouts to be polite and robust.

---

## File/Function Tour

- `LIST_URL`: FinCEN listing page.
- `UA`: Custom User‑Agent for requests.
- `get(url, stream=False)`: Centralized HTTP GET with headers, timeout, redirects, and error raising.
- `soup(url)`: Fetch + parse HTML via BeautifulSoup.
- `looks_like_pdf(url)`: Quick URL heuristic for `.pdf` (ignores query strings).
- `is_pdf_response(resp)`: Validates `Content-Type` contains `application/pdf`.
- `iter_non_spanish_fin_links(list_url)`: Yields eligible **FIN‑** advisory links from the listing.
- `find_first_pdf_on_page(page_url)`: Finds the first non‑Spanish PDF on an advisory page (prefers explicit non‑Spanish indicators; falls back to probing).
- `filename_from_response(resp, fallback_url)`: Chooses filename from `Content-Disposition` when present; otherwise uses the URL basename.
- `download_pdf(pdf_url, out_dir)`: Streams and saves the PDF; handles duplicates and collisions.
- `normalize_url(u)`: Removes URL fragments to prevent duplicate downloads.
- `main()`: Argument parsing, control loop (`--max`, `--out-dir`), logging, and error handling.

---

## Examples

**Single PDF (original behavior):**
```bash
python fincen_grabber.py
```

**Batch download:**
```bash
python fincen_grabber.py --max 10 --out-dir ./fincen_advisories
```

**No‑op (sanity check):**
```bash
python fincen_grabber.py --max 0
```

---

## Troubleshooting & Tips

- **No downloads happened**:  
  The page structure may have changed, or there were no eligible English `FIN-` advisories. The script prints a reason when it skips/fails a link.
- **“URL is not a PDF”**:  
  The link looked like a PDF but returned a non‑PDF content type. The script prevents saving the wrong file type.
- **Connection/timeout errors**:  
  Re‑run; transient network issues or site throttling can occur. The script continues past individual failures.
- **Spanish‑only advisories**:  
  By design, the script **excludes** Spanish versions, following your original requirement.
- **Respect robots and servers**:  
  Although this is light scraping, avoid excessive frequency.

---

## Legal/Ethical Considerations

- Content is from a U.S. government site; still, **use responsibly** and **cache locally** if running repeatedly.
- Do not misrepresent the files you download; keep advisory context and dates intact.

---

## Changelog

- **1.4**: Adds `--max` (batch downloads), `--out-dir`, duplicate skipping, filename collision handling, and improved robustness while preserving original logic.

---

## License

This script is provided **as‑is** without warranty. Review and adapt to your organization’s compliance needs.
