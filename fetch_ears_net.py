"""
fetch_ears_net.py
-----------------
Monitors the ECDC publication page for new EARS-Net annual data releases,
downloads the data file when a new release is detected, preprocesses it
into amr_tidy.csv format, and triggers a SQLite reload.

EARS-Net data is published annually every November. This script detects
new releases by comparing the latest publication date against the last
known release, so it can be safely scheduled to run weekly or daily
without redundant downloads.

Usage:
    python3 fetch_ears_net.py                  # check + download if new data
    python3 fetch_ears_net.py --force          # force re-download + reload
    python3 fetch_ears_net.py --check-only     # print status, no download
    python3 fetch_ears_net.py --dry-run        # download but don't write to DB

Schedule (weekly, every Monday 6am):
    0 6 * * 1 /path/to/venv/bin/python3 /path/to/fetch_ears_net.py >> /path/to/logs/ears_net.log 2>&1
"""

import os
import re
import sys
import json
import hashlib
import logging
import argparse
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ears_net_fetcher")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent
DATA_DIR      = _ROOT / "data"
OUTPUT_CSV    = DATA_DIR / "amr_tidy.csv"
STATE_FILE    = DATA_DIR / ".ears_net_state.json"   # tracks last seen release
BACKUP_DIR    = DATA_DIR / "backups"

DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# ── ECDC surveillance data page ───────────────────────────────────────────────
# This page lists all annual AMR surveillance reports with download links
ECDC_AMR_PAGE = (
    "https://www.ecdc.europa.eu/en/antimicrobial-resistance/surveillance-and-disease-data"
)

# Fallback: direct link to the most recent known data file
# ECDC publishes a structured data file alongside each annual report
ECDC_DATA_URLS = [
    # 2024 data (published Nov 2025) — update this each year
    "https://www.ecdc.europa.eu/sites/default/files/documents/surveillance-antimicrobial-resistance-europe-2024-data.xlsx",
    "https://www.ecdc.europa.eu/sites/default/files/documents/Surveillance-AMR-Europe-2024-data.xlsx",
    # 2023 data fallback
    "https://www.ecdc.europa.eu/sites/default/files/documents/surveillance-antimicrobial-resistance-europe-2023-data.xlsx",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AMR-Sentinel-Fetcher/1.0; "
        "research use; contact: amr-sentinel@research.edu)"
    )
}

# ── Column standardisation map ────────────────────────────────────────────────
# ECDC Excel files use varying column names across years — map all to standard
COLUMN_ALIASES = {
    # Country
    "country":          "country",
    "regionname":       "country",
    "countryname":      "country",
    "region":           "country",
    # Year
    "year":             "year",
    "time":             "year",
    "reportingyear":    "year",
    # Organism
    "organism":         "organism",
    "bacteria":         "organism",
    "pathogen":         "organism",
    "microorganism":    "organism",
    # Antibiotic
    "antibiotic":       "antibiotic",
    "antimicrobialagent": "antibiotic",
    "antibioticgroup":  "antibiotic",
    "antibioticcode":   "antibiotic",
    # Resistance %
    "pct_resistant":        "pct_resistant",
    "percentresistant":     "pct_resistant",
    "resistantpercent":     "pct_resistant",
    "numvalue":             "pct_resistant",
    "value":                "pct_resistant",
    "percentageresistant":  "pct_resistant",
    # Total isolates
    "total_isolates":       "total_isolates",
    "totalnum":             "total_isolates",
    "denominator":          "total_isolates",
    "numberoftested":       "total_isolates",
    "isolatestested":       "total_isolates",
}

EU_AGGREGATES = {"EU/EEA", "EU", "EEA", "European Union", "EU/EEA (total)"}


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_release_date": None, "last_checksum": None, "last_fetched": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Publication page scraper ──────────────────────────────────────────────────

def scrape_latest_release() -> dict | None:
    """
    Scrape the ECDC AMR surveillance page to find the most recent
    annual data publication and its date.

    Returns dict with keys: title, date, url — or None on failure.
    """
    try:
        log.info(f"Checking ECDC publication page for new releases ...")
        r = requests.get(ECDC_AMR_PAGE, headers=HEADERS, timeout=30)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Look for links containing "surveillance" and "antimicrobial-resistance"
        # and "data" — these are the annual data files
        candidates = []
        for link in soup.find_all("a", href=True):
            href  = link["href"]
            text  = link.get_text(strip=True)
            # Match patterns like "Surveillance of antimicrobial resistance in Europe, 2024 data"
            if re.search(r"antimicrobial.resistance.*\d{4}.*data|AMR.*\d{4}.*data", text, re.I):
                # Try to extract year from text
                year_match = re.search(r"20\d{2}", text)
                year = int(year_match.group()) if year_match else 0
                candidates.append({
                    "title": text,
                    "year":  year,
                    "url":   href if href.startswith("http") else f"https://www.ecdc.europa.eu{href}",
                })

        if not candidates:
            log.warning("No AMR data publications found on ECDC page — using fallback URLs")
            return None

        # Return the most recent by year
        latest = max(candidates, key=lambda x: x["year"])
        log.info(f"Latest release found: {latest['title']} ({latest['year']})")
        return latest

    except Exception as e:
        log.warning(f"Failed to scrape ECDC page: {e}")
        return None


def find_download_url(publication_url: str) -> str | None:
    """
    Visit a publication page and find the direct Excel/CSV download link.
    """
    try:
        r = requests.get(publication_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if re.search(r"\.(xlsx|xls|csv)$", href, re.I):
                return href if href.startswith("http") else f"https://www.ecdc.europa.eu{href}"

        return None
    except Exception as e:
        log.warning(f"Failed to find download URL at {publication_url}: {e}")
        return None


# ── Downloader ────────────────────────────────────────────────────────────────

def download_file(url: str) -> bytes | None:
    """Download a file from URL, return raw bytes."""
    try:
        log.info(f"Downloading: {url}")
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        content = r.content
        log.info(f"Downloaded {len(content):,} bytes")
        return content
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def try_fallback_urls() -> bytes | None:
    """Try the hardcoded fallback URLs in order."""
    for url in ECDC_DATA_URLS:
        log.info(f"Trying fallback URL: {url}")
        content = download_file(url)
        if content and len(content) > 10_000:  # must be a real file, not an error page
            return content
    return None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using COLUMN_ALIASES."""
    rename = {}
    for col in df.columns:
        key = col.lower().strip().replace(" ", "").replace("_", "").replace("-", "")
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def parse_excel(content: bytes) -> pd.DataFrame | None:
    """
    Parse the ECDC Excel file. Try each sheet to find the one with
    resistance percentage data.
    """
    try:
        xl = pd.ExcelFile(pd.io.common.BytesIO(content))
        log.info(f"Excel sheets: {xl.sheet_names}")

        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, header=0)
            df = normalise_columns(df)

            # Check if this sheet has the columns we need
            has_resistance = "pct_resistant" in df.columns
            has_country    = "country" in df.columns
            has_organism   = "organism" in df.columns

            if has_resistance and has_country:
                log.info(f"Using sheet '{sheet}' ({len(df)} rows)")
                return df

        log.warning("No suitable sheet found in Excel file")
        return None

    except Exception as e:
        log.error(f"Failed to parse Excel: {e}")
        return None


def parse_csv(content: bytes) -> pd.DataFrame | None:
    """Parse CSV content."""
    try:
        text = content.decode("utf-8-sig")
        df   = pd.read_csv(StringIO(text))
        df   = normalise_columns(df)
        return df
    except Exception as e:
        log.error(f"Failed to parse CSV: {e}")
        return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise into amr_tidy.csv format:
    - Ensure required columns exist
    - Drop EU aggregate rows
    - Filter invalid resistance values
    - Sort consistently
    """
    required = ["country", "pct_resistant"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing after normalisation. "
                             f"Available: {list(df.columns)}")

    df = df.copy()

    # Fill optional columns with defaults if missing
    if "year"          not in df.columns: df["year"]          = datetime.now().year - 1
    if "organism"      not in df.columns: df["organism"]      = "Unknown"
    if "antibiotic"    not in df.columns: df["antibiotic"]    = "Unknown"
    if "total_isolates" not in df.columns: df["total_isolates"] = 0

    # Type coercion
    df["pct_resistant"]  = pd.to_numeric(df["pct_resistant"],  errors="coerce")
    df["total_isolates"] = pd.to_numeric(df["total_isolates"], errors="coerce").fillna(0).astype(int)
    df["year"]           = pd.to_numeric(df["year"],           errors="coerce").fillna(0).astype(int)

    # Drop EU aggregates and invalid rows
    df = df[~df["country"].isin(EU_AGGREGATES)]
    df = df.dropna(subset=["pct_resistant"])
    df = df[df["pct_resistant"].between(0, 100)]
    df = df[df["year"] > 2000]   # sanity check

    # Keep only standard columns
    df = df[["country", "year", "organism", "antibiotic", "pct_resistant", "total_isolates"]]
    df = df.sort_values(["organism", "antibiotic", "country", "year"]).reset_index(drop=True)

    return df


# ── Change detection ──────────────────────────────────────────────────────────

def checksum(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


# ── Backup ────────────────────────────────────────────────────────────────────

def backup(csv_path: Path):
    if csv_path.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"amr_tidy_{ts}.csv"
        backup.write_bytes(csv_path.read_bytes())
        log.info(f"Backup saved → {backup.name}")


# ── SQLite reload ─────────────────────────────────────────────────────────────

def reload_sqlite(csv_path: Path):
    try:
        sys.path.insert(0, str(_ROOT))
        import db_setup
        if hasattr(db_setup, "ingest_csv"):
            log.info("Reloading SQLite ...")
            db_setup.ingest_csv(str(csv_path))
            log.info("SQLite reload complete.")
        else:
            log.warning("db_setup.ingest_csv() not found — run db_setup.py manually.")
    except Exception as e:
        log.error(f"SQLite reload failed: {e}")
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EARS-Net automated data fetcher")
    parser.add_argument("--force",      action="store_true", help="Force re-download even if unchanged")
    parser.add_argument("--check-only", action="store_true", help="Check for new data only, no download")
    parser.add_argument("--dry-run",    action="store_true", help="Download + preprocess but don't write")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  EARS-Net Fetcher — AMR Sentinel")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    state = load_state()

    # ── Step 1: Check for new release ────────────────────────────────
    latest   = scrape_latest_release()
    new_year = latest["year"] if latest else None
    last_year = state.get("last_release_year")

    if new_year and last_year and new_year <= last_year and not args.force:
        log.info(f"No new data — last fetched: {last_year}, latest available: {new_year}")
        if args.check_only:
            log.info("Up to date. Run with --force to reload anyway.")
        sys.exit(0)

    if new_year and new_year > (last_year or 0):
        log.info(f"New data detected: {new_year} (previously had {last_year})")
    elif args.force:
        log.info("--force flag set — downloading regardless.")

    if args.check_only:
        log.info(f"New data available ({new_year}). Run without --check-only to download.")
        sys.exit(0)

    # ── Step 2: Find download URL ─────────────────────────────────────
    content = None

    if latest and latest.get("url"):
        # Try to find direct download link from publication page
        dl_url = find_download_url(latest["url"])
        if dl_url:
            content = download_file(dl_url)

    # Fallback to hardcoded URLs if scraping didn't find a file
    if not content or len(content) < 10_000:
        log.info("Trying fallback download URLs ...")
        content = try_fallback_urls()

    if not content:
        log.error("All download attempts failed. Check network or update ECDC_DATA_URLS in script.")
        sys.exit(1)

    # ── Step 3: Parse ─────────────────────────────────────────────────
    # Detect file type from content
    if content[:4] in (b'PK\x03\x04', b'\xd0\xcf\x11\xe0'):  # ZIP (xlsx) or OLE (xls)
        df_raw = parse_excel(content)
    else:
        df_raw = parse_csv(content)

    if df_raw is None:
        log.error("Failed to parse downloaded file.")
        sys.exit(1)

    # ── Step 4: Preprocess ────────────────────────────────────────────
    try:
        df = preprocess(df_raw)
    except ValueError as e:
        log.error(f"Preprocessing failed: {e}")
        log.error("The ECDC file format may have changed — update COLUMN_ALIASES in the script.")
        sys.exit(1)

    log.info(
        f"Preprocessed: {len(df):,} rows | "
        f"{df['organism'].nunique()} organisms | "
        f"{df['country'].nunique()} countries | "
        f"years {df['year'].min()}–{df['year'].max()}"
    )

    # ── Step 5: Change detection ──────────────────────────────────────
    new_checksum = checksum(df)
    if new_checksum == state.get("last_checksum") and not args.force:
        log.info("Data content unchanged — skipping write.")
        sys.exit(0)

    if args.dry_run:
        log.info("--dry-run: not writing. Preview:")
        print(df.head(10).to_string())
        sys.exit(0)

    # ── Step 6: Backup + write ────────────────────────────────────────
    backup(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Written → {OUTPUT_CSV}")

    # ── Step 7: Reload SQLite ─────────────────────────────────────────
    reload_sqlite(OUTPUT_CSV)

    # ── Step 8: Save state ────────────────────────────────────────────
    state.update({
        "last_release_year": new_year,
        "last_checksum":     new_checksum,
        "last_fetched":      datetime.now().isoformat(),
        "rows":              len(df),
        "organisms":         df["organism"].nunique(),
        "countries":         df["country"].nunique(),
        "year_range":        [int(df["year"].min()), int(df["year"].max())],
    })
    save_state(state)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Update complete.")
    log.info(f"  Rows:      {len(df):,}")
    log.info(f"  Organisms: {df['organism'].nunique()}")
    log.info(f"  Countries: {df['country'].nunique()}")
    log.info(f"  Years:     {df['year'].min()} – {df['year'].max()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
