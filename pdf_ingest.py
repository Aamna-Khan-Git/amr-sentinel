"""
pdf_ingest.py
-------------
Downloads the ECDC EARS-Net Annual Epidemiological Report PDF,
extracts structured resistance percentage tables, and ingests data into:
  1. SQLite  (structured resistance %, queryable by organism/antibiotic/country/year)
  2. ChromaDB (narrative text chunks, for semantic search by literature_agent)

Also updates fetch_ears_net state so the monitor knows data is current.

Usage:
    python3 pdf_ingest.py                        # download latest + ingest
    python3 pdf_ingest.py --pdf path/to/file.pdf # ingest from local PDF
    python3 pdf_ingest.py --year 2023            # download a specific year
    python3 pdf_ingest.py --dry-run              # extract but don't write
"""

import re
import sys
import json
import logging
import argparse
import hashlib
import requests
import pdfplumber
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdf_ingest")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent
DATA_DIR   = _ROOT / "data"
OUTPUT_CSV = DATA_DIR / "amr_tidy.csv"
STATE_FILE = DATA_DIR / ".ears_net_state.json"
BACKUP_DIR = DATA_DIR / "backups"
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# ── PDF URLs per year ─────────────────────────────────────────────────────────
PDF_URLS = {
    2024: "https://www.ecdc.europa.eu/sites/default/files/documents/antimicrobial-resistance-eu-annual-epidemiological-report-2024.pdf",
    2023: "https://www.ecdc.europa.eu/sites/default/files/documents/antimicrobial-resistance-annual-epidemiological-report-EARS-Net-2023.pdf",
    2022: "https://www.ecdc.europa.eu/sites/default/files/documents/antimicrobial-resistance-annual-epidemiological-report-EARS-Net-2022.pdf",
}

HEADERS = {
    "User-Agent":      "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate",  # exclude brotli to avoid decode errors
}

# ── Organism name normalisation ───────────────────────────────────────────────
ORG_MAP = {
    "escherichia coli":       "E. coli",
    "e. coli":                "E. coli",
    "klebsiella pneumoniae":  "K. pneumoniae",
    "pseudomonas aeruginosa": "P. aeruginosa",
    "acinetobacter":          "A. baumannii",
    "acinetobacter species":  "A. baumannii",
    "staphylococcus aureus":  "S. aureus",
    "streptococcus pneumoniae": "S. pneumoniae",
    "enterococcus faecalis":  "E. faecalis",
    "enterococcus faecium":   "E. faecium",
}

# Antibiotic group → short code
AB_MAP = {
    "aminopenicillin":             "AMP",
    "amoxicillin":                 "AMP",
    "ampicillin":                  "AMP",
    "third-generation cephalosporin": "C3G",
    "cefotaxime":                  "C3G",
    "ceftriaxone":                 "C3G",
    "carbapenem":                  "CAR",
    "imipenem":                    "CAR",
    "meropenem":                   "CAR",
    "fluoroquinolone":             "CIP",
    "ciprofloxacin":               "CIP",
    "aminoglycoside":              "GEN",
    "gentamicin":                  "GEN",
    "colistin":                    "COL",
    "vancomycin":                  "VAN",
    "mrsa":                        "OXA",
    "meticillin":                  "OXA",
    "oxacillin":                   "OXA",
    "piperacillin":                "TZP",
    "esbl":                        "ESBL_PHENO",
    "combined resistance":         "COMBINED",
    "multidrug":                   "COMBINED",
}


def normalise_organism(text: str) -> str:
    t = text.lower().strip()
    for key, val in ORG_MAP.items():
        if key in t:
            return val
    return text.strip()


def normalise_antibiotic(text: str) -> str:
    t = text.lower()
    for key, val in AB_MAP.items():
        if key in t:
            return val
    # Return first word uppercased as fallback
    return text.split()[0].upper() if text.strip() else "UNKNOWN"


# ── PDF downloader ────────────────────────────────────────────────────────────

def download_pdf(year: int) -> bytes:
    url = PDF_URLS.get(year)
    if not url:
        raise ValueError(f"No URL configured for year {year}. Add it to PDF_URLS.")
    log.info(f"Downloading {year} report: {url}")
    r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    data = b"".join(r.iter_content(chunk_size=8192))
    log.info(f"Downloaded {len(data):,} bytes")
    return data


# ── Table extractor ───────────────────────────────────────────────────────────

def is_resistance_table(table: list) -> bool:
    """
    Check if a table contains resistance percentage data.
    Key signals: columns with %, 'resistance', organism names, year columns.
    """
    flat = " ".join(
        str(cell).lower()
        for row in table[:3]
        for cell in row
        if cell
    )
    has_resistance = "resistance" in flat or "resistant" in flat
    has_percent    = "%" in flat or any(
        re.search(r"\d+\.\d", str(cell) or "")
        for row in table[1:4]
        for cell in row
    )
    has_organism   = any(
        org in flat
        for org in ["escherichia", "klebsiella", "staphylococcus",
                    "enterococcus", "pseudomonas", "acinetobacter",
                    "streptococcus"]
    )
    return has_resistance and has_percent and has_organism


def extract_rows_from_table(table: list, data_year: int) -> list[dict]:
    """
    Parse a resistance table into structured rows.
    Handles the two main formats found in ECDC PDFs:
      Format A: organism | antibiotic | n | % | n | % ...  (multi-year)
      Format B: organism | antibiotic | EU/EEA range | single year %
    """
    rows = []
    current_organism = None

    # Find header row — look for year columns or n/% columns
    header_idx = 0
    for i, row in enumerate(table[:4]):
        flat = " ".join(str(c) for c in row if c).lower()
        if "%" in flat or "2020" in flat or "2021" in flat:
            header_idx = i
            break

    header = table[header_idx] if header_idx < len(table) else []

    # Find year columns in header
    year_col_map = {}  # col_idx → year
    for idx, cell in enumerate(header):
        if cell:
            m = re.search(r"(20\d{2})", str(cell))
            if m:
                year_col_map[idx] = int(m.group(1))

    # Find % columns (look for cells containing just digits.digits in data rows)
    # Strategy: collect all numeric columns from first data rows
    data_start = header_idx + 1
    if data_start >= len(table):
        return rows

    for row in table[data_start:]:
        if not any(row):
            continue

        # Track current organism across merged cells
        first_cell = str(row[0] or "").strip()
        if first_cell and len(first_cell) > 3:
            candidate = normalise_organism(first_cell)
            if candidate in ORG_MAP.values():
                current_organism = candidate
            elif any(org in first_cell.lower() for org in
                     ["escherichia", "klebsiella", "staphylococcus",
                      "enterococcus", "pseudomonas", "acinetobacter",
                      "streptococcus"]):
                current_organism = normalise_organism(first_cell)

        if not current_organism:
            continue

        # Find antibiotic description column — usually col 1 or 3
        antibiotic_text = ""
        for idx in [1, 2, 3]:
            if idx < len(row) and row[idx]:
                cell_text = str(row[idx]).strip()
                if len(cell_text) > 5 and not re.match(r"^\d", cell_text):
                    antibiotic_text = cell_text
                    break

        if not antibiotic_text:
            continue

        ab_code = normalise_antibiotic(antibiotic_text)

        # Extract year → % pairs
        # Approach: scan all cells for float values that look like percentages (0-100)
        # and pair with nearest year header
        numeric_cells = []
        for idx, cell in enumerate(row):
            if cell is None:
                continue
            val = re.sub(r"[^\d.]", "", str(cell))
            try:
                f = float(val)
                if 0.0 <= f <= 100.0 and "." in str(cell):
                    numeric_cells.append((idx, f))
            except ValueError:
                continue

        # Try to match numeric cells to years from header
        for col_idx, pct in numeric_cells:
            # Find closest year header
            year = None
            if year_col_map:
                closest = min(year_col_map.keys(),
                              key=lambda k: abs(k - col_idx),
                              default=None)
                if closest is not None and abs(closest - col_idx) <= 3:
                    year = year_col_map[closest]

            if not year:
                year = data_year  # fallback to report year

            rows.append({
                "country":        "EU/EEA",
                "year":           year,
                "organism":       current_organism,
                "antibiotic":     ab_code,
                "pct_resistant":  round(pct, 2),
                "total_isolates": 0,
                "source":         "ECDC_PDF",
            })

    return rows


def extract_text_chunks(pdf) -> list[str]:
    """Extract narrative text chunks for ChromaDB ingestion."""
    chunks = []
    for page in pdf.pages:
        text = page.extract_text()
        if text and len(text) > 100:
            # Split into ~500 char chunks at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", text)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) > 500:
                    if chunk:
                        chunks.append(chunk.strip())
                    chunk = s
                else:
                    chunk += " " + s
            if chunk:
                chunks.append(chunk.strip())
    return chunks


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_from_pdf(pdf_bytes: bytes, report_year: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Open PDF, find resistance tables, extract rows.
    Returns (DataFrame, list of text chunks).
    """
    all_rows   = []
    all_chunks = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        log.info(f"PDF has {len(pdf.pages)} pages")

        for i, page in enumerate(pdf.pages):
            # Extract text for ChromaDB
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                all_chunks.append(f"[Page {i+1}] {text.strip()}")

            # Extract tables for SQLite
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 3:
                    continue
                if is_resistance_table(table):
                    rows = extract_rows_from_table(table, report_year)
                    if rows:
                        log.info(f"  Page {i+1}: extracted {len(rows)} rows")
                        all_rows.extend(rows)

    if not all_rows:
        log.warning("No resistance rows extracted — check is_resistance_table() logic")
        return pd.DataFrame(), all_chunks

    df = pd.DataFrame(all_rows)

    # Deduplicate
    df = df.drop_duplicates(
        subset=["country", "year", "organism", "antibiotic", "pct_resistant"]
    )

    # Sort
    df = df.sort_values(
        ["organism", "antibiotic", "country", "year"]
    ).reset_index(drop=True)

    log.info(f"Total extracted: {len(df):,} rows | "
             f"{df['organism'].nunique()} organisms | "
             f"{df['year'].nunique()} years")

    return df, all_chunks


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


# ── ChromaDB ingest ───────────────────────────────────────────────────────────

def ingest_chromadb(chunks: list[str], report_year: int):
    """Ingest text chunks into ChromaDB via literature_store.py."""
    try:
        sys.path.insert(0, str(_ROOT))
        import literature_store

        log.info(f"Ingesting {len(chunks)} text chunks into ChromaDB ...")

        # Build records compatible with literature_store's expected format
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "pmid":     f"ECDC_{report_year}_{i:04d}",
                "title":    f"ECDC EARS-Net Annual Report {report_year}",
                "abstract": chunk,
                "year":     report_year,
                "source":   "ECDC_PDF",
            })

        if hasattr(literature_store, "ingest_records"):
            literature_store.ingest_records(records)
        elif hasattr(literature_store, "upsert"):
            literature_store.upsert(records)
        else:
            log.warning(
                "literature_store has no ingest_records() or upsert() — "
                "add one of these functions or ingest manually."
            )
            return

        log.info("ChromaDB ingest complete.")

    except ImportError:
        log.warning("literature_store.py not found — skipping ChromaDB ingest.")
    except Exception as e:
        log.error(f"ChromaDB ingest failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECDC PDF → SQLite + ChromaDB ingestion")
    parser.add_argument("--pdf",     type=str, help="Path to local PDF file")
    parser.add_argument("--year",    type=int, default=2024, help="Report year (default: 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't write")
    parser.add_argument("--no-chroma", action="store_true", help="Skip ChromaDB ingest")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  ECDC PDF Ingestion Pipeline — AMR Sentinel")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # ── Step 1: Get PDF bytes ─────────────────────────────────────────
    if args.pdf:
        path = Path(args.pdf)
        if not path.exists():
            log.error(f"File not found: {path}")
            sys.exit(1)
        log.info(f"Using local PDF: {path}")
        pdf_bytes = path.read_bytes()
    else:
        pdf_bytes = download_pdf(args.year)

    # ── Step 2: Extract ───────────────────────────────────────────────
    df, chunks = extract_from_pdf(pdf_bytes, args.year)

    if df.empty:
        log.error("No data extracted from PDF.")
        sys.exit(1)

    log.info(f"\nExtracted data preview:")
    print(df.head(10).to_string())
    print()

    if args.dry_run:
        log.info("--dry-run: not writing to disk.")
        sys.exit(0)

    # ── Step 3: Merge with existing CSV ──────────────────────────────
    # Keep existing data and add/update with new PDF data
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        log.info(f"Existing CSV: {len(existing):,} rows")

        # Remove old PDF-sourced rows for this year to avoid duplicates
        if "source" in existing.columns:
            existing = existing[
                ~((existing["source"] == "ECDC_PDF") &
                  (existing["year"] == args.year))
            ]

        # Keep source column consistent
        if "source" not in existing.columns:
            existing["source"] = "ORIGINAL"

        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["country", "year", "organism", "antibiotic", "pct_resistant"]
        )
    else:
        combined = df

    combined = combined.sort_values(
        ["organism", "antibiotic", "country", "year"]
    ).reset_index(drop=True)

    log.info(f"Combined CSV: {len(combined):,} rows")

    # ── Step 4: Backup + write ────────────────────────────────────────
    if OUTPUT_CSV.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"amr_tidy_{ts}.csv"
        backup.write_bytes(OUTPUT_CSV.read_bytes())
        log.info(f"Backup → {backup.name}")

    combined.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Written → {OUTPUT_CSV}")

    # ── Step 5: Reload SQLite ─────────────────────────────────────────
    reload_sqlite(OUTPUT_CSV)

    # ── Step 6: ChromaDB ─────────────────────────────────────────────
    if not args.no_chroma and chunks:
        ingest_chromadb(chunks, args.year)

    # ── Step 7: Update state ──────────────────────────────────────────
    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())

    state.update({
        "last_reload":            datetime.now().isoformat(),
        "last_known_release_year": args.year,
        "rows":                   len(combined),
        "year_range":             [int(combined["year"].min()),
                                   int(combined["year"].max())],
        "source":                 "ECDC_PDF",
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))

    log.info("=" * 60)
    log.info(f"  Done. {len(combined):,} total rows in database.")
    log.info(f"  Years: {combined['year'].min()} – {combined['year'].max()}")
    log.info(f"  Organisms: {combined['organism'].nunique()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
