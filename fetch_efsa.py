"""
fetch_efsa.py
-------------
Downloads EFSA/ECDC joint AMR zoonotic surveillance data and ingests it
into the AMR Sentinel SQLite database alongside ECDC human bloodstream data.

Data covers:
  - Salmonella spp. in humans (clinical cases)
  - Salmonella spp. in animals (pigs, broilers, cattle)
  - Salmonella spp. in meat (pork, broiler meat)
  - E. coli in animals (pigs, broilers, cattle)
  - E. coli in humans

This enables One Health queries spanning human, animal, and food chain
resistance patterns within the same system.

Data sources (open access via Zenodo):
  - 2015 data: https://zenodo.org/records/495574
  (Email zoonoses_support@efsa.europa.eu for more recent structured data)

Usage:
    python3 fetch_efsa.py                      # ingest all available data
    python3 fetch_efsa.py --dry-run            # preview without writing
    python3 fetch_efsa.py --xlsx path/to/file  # ingest from local file
"""

import re
import sys
import json
import time
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fetch_efsa")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent
DATA_DIR   = _ROOT / "data"
OUTPUT_CSV = DATA_DIR / "amr_tidy.csv"
BACKUP_DIR = DATA_DIR / "backups"
STATE_FILE = DATA_DIR / ".efsa_state.json"
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# ── Data sources ──────────────────────────────────────────────────────────────
EFSA_SOURCES = [
    {
        "year": 2015,
        "url":  "https://zenodo.org/records/495574/files/Data%20Viz%202015%20updates_ECDC.xlsx?download=1",
        "description": "EFSA/ECDC joint AMR report 2015 — Salmonella, E.coli, ESBL",
    },
    # Add newer files here as they become available from EFSA
    # {
    #     "year": 2023,
    #     "url":  "...",  # request from zoonoses_support@efsa.europa.eu
    # },
]

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (AMR-Sentinel research tool)",
    "Accept-Encoding": "gzip, deflate",
}

# ── Downloader ────────────────────────────────────────────────────────────────

def download_with_retry(url: str, retries: int = 3, backoff: int = 10) -> bytes:
    """Download a file with automatic retry on timeout or server errors."""
    for attempt in range(1, retries + 1):
        try:
            log.info(f"Downloading (attempt {attempt}/{retries}): {url}")
            r = requests.get(url, headers=HEADERS, timeout=90, stream=True)
            r.raise_for_status()
            data = b"".join(r.iter_content(8192))
            log.info(f"Downloaded {len(data):,} bytes")
            return data
        except Exception as e:
            log.warning(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                log.info(f"Retrying in {backoff}s ...")
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"All {retries} download attempts failed for {url}")


# ── Sheet parsers ─────────────────────────────────────────────────────────────
# Each parser handles one sheet's messy multi-header format

EU_AGGREGATES = {"EU", "EU/EEA", "European Union", "EU total", "EU average"}

# Antibiotic name → standard code
AB_ALIASES = {
    "ampicillin":          "AMP",
    "amoxicillin":         "AMP",
    "tetracycline":        "TET",
    "ciprofloxacin":       "CIP",
    "nalidixic":           "NAL",
    "chloramphenicol":     "CHL",
    "gentamicin":          "GEN",
    "streptomycin":        "STR",
    "trimethoprim":        "TMP",
    "sulfamethoxazole":    "SUL",
    "sulfonamide":         "SUL",
    "cefotaxime":          "C3G",
    "ceftriaxone":         "C3G",
    "cephalosporin":       "C3G",
    "carbapenem":          "CAR",
    "imipenem":            "CAR",
    "meropenem":           "CAR",
    "colistin":            "COL",
    "vancomycin":          "VAN",
    "esbl":                "ESBL_PHENO",
    "amc":                 "ESBL_PHENO",  # AmC = AmpC in EFSA notation
}

def get_ab_code(col_name: str) -> str:
    t = col_name.lower()
    for key, code in AB_ALIASES.items():
        if key in t:
            return code
    # Return cleaned column name as fallback
    return re.sub(r"[^a-z0-9]", "", t[:10]).upper()


def parse_resistance_sheet(
    df: pd.DataFrame,
    organism: str,
    source_type: str,
    year: int,
) -> list[dict]:
    """
    Parse EFSA resistance sheets. Two formats observed:

    Format A (E.coli, Salmonella sheets):
      Row 0: title (ignore)
      Row 1: antibiotic names  [Country, Ampicillin, Azithromycin, ...]
      Row 2: %pos headers      [NaN,     %pos,       %pos,         ...]
      Row 3+: data             [Austria, 12.9,       0,            ...]

    Format B (Salm H sheet):
      Row 0: title (ignore)
      Row 1: antibiotic names  [Country, Ampicillin, NaN,   Azithromycin, NaN, ...]
      Row 2: N/%Res headers    [NaN,     N,          %Res,  N,            %Res ...]
      Row 3+: data             [Austria, 1556,       13.5,  -, -,         ...]
    """
    rows = []

    if len(df) < 4:
        return rows

    # Row 0 = title, Row 1 = antibiotic names, Row 2 = N/%pos/%Res, Row 3+ = data
    ab_row   = df.iloc[1].tolist()   # antibiotic names
    pct_row  = df.iloc[2].tolist()   # %pos / % Res / N markers
    data_start = 3

    # Build map: col_idx → (ab_code, n_col_idx or None)
    # Walk ab_row to find antibiotic names
    ab_col_map  = {}   # pct_col_idx → ab_code
    n_col_map   = {}   # pct_col_idx → n_col_idx

    current_ab   = None
    current_n    = None

    for idx, ab_cell in enumerate(ab_row):
        ab_str = str(ab_cell or "").strip()

        # New antibiotic name found
        if (len(ab_str) > 2
                and not re.search(r"^(nan|country|n|%|note)$", ab_str, re.I)
                and not ab_str.startswith("Unnamed")):
            current_ab = get_ab_code(ab_str)
            current_n  = None

        if current_ab is None:
            continue

        pct_str = str(pct_row[idx] if idx < len(pct_row) else "").strip().lower()

        if re.search(r"^n$|^num", pct_str):
            current_n = idx
        elif re.search(r"%\s*(res|pos|r\b)|%res|%pos", pct_str):
            ab_col_map[idx] = current_ab
            if current_n is not None:
                n_col_map[idx] = current_n

    if not ab_col_map:
        log.debug(f"No resistance columns found in {organism}/{source_type}")
        return rows

    # Extract data rows
    for i in range(data_start, len(df)):
        row = df.iloc[i].tolist()

        country = str(row[0] or "").strip()
        # Clean up country name
        country = re.sub(r"\s*\(.*?\)", "", country).strip()  # remove "(a)" etc

        if not country or country.lower() in ("nan", "total", "average",
                                               "mean", "eu", "eu/eea"):
            continue
        if country in EU_AGGREGATES:
            continue
        if len(country) < 2:
            continue
        if re.search(r"^(note|source|data|refer)", country, re.I):
            continue

        for pct_col, ab_code in ab_col_map.items():
            if pct_col >= len(row):
                continue

            # Get resistance %
            pct_val = str(row[pct_col] or "").strip()
            pct_val = re.sub(r"[^\d.]", "", pct_val)  # strip (a), *, etc
            try:
                pct = float(pct_val)
                if not (0.0 <= pct <= 100.0):
                    continue
            except (ValueError, TypeError):
                continue

            # Get isolate count if available
            n_isolates = 0
            if pct_col in n_col_map:
                n_col = n_col_map[pct_col]
                if n_col < len(row):
                    n_val = str(row[n_col] or "").strip()
                    n_val = re.sub(r"[^\d]", "", n_val)
                    try:
                        n_isolates = int(n_val) if n_val else 0
                    except ValueError:
                        n_isolates = 0

            rows.append({
                "country":        country,
                "year":           year,
                "organism":       organism,
                "antibiotic":     ab_code,
                "pct_resistant":  round(pct, 2),
                "total_isolates": n_isolates,
                "source_type":    source_type,
                "source":         "EFSA",
            })

    return rows


# ── Main xlsx parser ──────────────────────────────────────────────────────────
    """
    Parse the full EFSA Excel file across all relevant sheets.
    Returns a combined DataFrame.
    """
    xl       = pd.ExcelFile(BytesIO(content))
    all_rows = []

    log.info(f"Sheets: {xl.sheet_names}")

    sheet_config = {
        # sheet_name_pattern → (organism, source_type)
        "e.coli":     ("E. coli",     "animal"),
        "ecoli":      ("E. coli",     "animal"),
        "salmonella": ("Salmonella spp.", "mixed"),
        "salm h":     ("Salmonella spp.", "human"),
        "salm_h":     ("Salmonella spp.", "human"),
    }

    for sheet in xl.sheet_names:
        sheet_key = sheet.lower().strip()
        config    = None

        for pattern, (organism, source_type) in sheet_config.items():
            if pattern in sheet_key:
                config = (organism, source_type)
                break

        if not config:
            log.debug(f"Skipping sheet: {sheet}")
            continue

        organism, source_type = config

        # Refine source_type from sheet name
        s = sheet_key
        if "human" in s or "salm h" in s:
            source_type = "human"
        elif "pig" in s or "pork" in s:
            source_type = "animal_pig"
        elif "broil" in s or "poultry" in s or "chicken" in s:
            source_type = "animal_broiler"
        elif "cattle" in s or "bovine" in s or "calf" in s:
            source_type = "animal_cattle"
        elif "meat" in s:
            source_type = "meat"
        elif "animal" in s:
            source_type = "animal"

        log.info(f"Parsing sheet '{sheet}' → {organism} / {source_type}")

        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=None)

            # Refine source_type from the actual title in row 0
            title = str(df.iloc[0, 0] or "").lower()
            if "human" in title:
                source_type = "human"
            elif "meat" in title and "pig" in title:
                source_type = "meat_pork"
            elif "meat" in title and ("broil" in title or "poultry" in title):
                source_type = "meat_broiler"
            elif "meat" in title and ("cattle" in title or "bovine" in title):
                source_type = "meat_cattle"
            elif "meat" in title:
                source_type = "meat"
            elif "pig" in title or "pork" in title or "swine" in title:
                source_type = "animal_pig"
            elif "broil" in title or "poultry" in title or "chicken" in title:
                source_type = "animal_broiler"
            elif "cattle" in title or "bovine" in title or "calf" in title:
                source_type = "animal_cattle"
            elif "turkey" in title:
                source_type = "animal_turkey"
            elif "animal" in title:
                source_type = "animal"

            log.info(f"  Title: '{df.iloc[0, 0]}' → source_type={source_type}")
            rows = parse_resistance_sheet(df, organism, source_type, year)
            log.info(f"  Extracted {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            log.warning(f"  Failed to parse sheet '{sheet}': {e}")

    if not all_rows:
        log.warning("No rows extracted from EFSA file")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(
        subset=["country", "year", "organism", "antibiotic", "source_type"]
    )
    df = df.sort_values(
        ["organism", "source_type", "antibiotic", "country", "year"]
    ).reset_index(drop=True)

    return df


# ── Main xlsx parser ──────────────────────────────────────────────────────────

def parse_efsa_xlsx(content: bytes, year: int) -> pd.DataFrame:
    """Parse the full EFSA Excel file across all relevant sheets."""
    xl       = pd.ExcelFile(BytesIO(content))
    all_rows = []

    log.info(f"Sheets: {xl.sheet_names}")

    sheet_config = {
        "e.coli":     ("E. coli",        "animal"),
        "ecoli":      ("E. coli",        "animal"),
        "salmonella": ("Salmonella spp.", "mixed"),
        "salm h":     ("Salmonella spp.", "human"),
        "salm_h":     ("Salmonella spp.", "human"),
    }

    for sheet in xl.sheet_names:
        sheet_key = sheet.lower().strip()
        config    = None
        for pattern, (organism, source_type) in sheet_config.items():
            if pattern in sheet_key:
                config = (organism, source_type)
                break
        if not config:
            log.debug(f"Skipping sheet: {sheet}")
            continue

        organism, source_type = config

        try:
            df    = pd.read_excel(xl, sheet_name=sheet, header=None)
            title = str(df.iloc[0, 0] or "").lower()

            if "human" in title:
                source_type = "human"
            elif "meat" in title and "pig" in title:
                source_type = "meat_pork"
            elif "meat" in title and ("broil" in title or "poultry" in title):
                source_type = "meat_broiler"
            elif "meat" in title:
                source_type = "meat"
            elif "pig" in title or "pork" in title or "swine" in title:
                source_type = "animal_pig"
            elif "broil" in title or "poultry" in title or "chicken" in title:
                source_type = "animal_broiler"
            elif "cattle" in title or "bovine" in title:
                source_type = "animal_cattle"

            log.info(f"Parsing sheet '{sheet}' → {organism} / {source_type}")
            log.info(f"  Title: '{df.iloc[0, 0]}' → source_type={source_type}")

            rows = parse_resistance_sheet(df, organism, source_type, year)
            log.info(f"  Extracted {len(rows)} rows")
            all_rows.extend(rows)

        except Exception as e:
            log.warning(f"Failed to parse sheet '{sheet}': {e}")

    if not all_rows:
        log.warning("No rows extracted from EFSA file")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(
        subset=["country", "year", "organism", "antibiotic", "source_type"]
    )
    df = df.sort_values(
        ["organism", "source_type", "antibiotic", "country", "year"]
    ).reset_index(drop=True)
    return df


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EFSA AMR zoonotic data fetcher")
    parser.add_argument("--xlsx",    type=str, help="Path to local EFSA Excel file")
    parser.add_argument("--dry-run", action="store_true", help="Preview, don't write")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  EFSA Zoonotic AMR Fetcher — AMR Sentinel")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    all_frames = []

    if args.xlsx:
        # Local file provided
        path = Path(args.xlsx)
        if not path.exists():
            log.error(f"File not found: {path}")
            sys.exit(1)
        log.info(f"Using local file: {path}")
        # Guess year from filename
        year_match = re.search(r"20\d{2}", path.name)
        year = int(year_match.group()) if year_match else 2015
        content = path.read_bytes()
        df = parse_efsa_xlsx(content, year)
        if not df.empty:
            all_frames.append(df)
    else:
        # Download from configured sources
        for source in EFSA_SOURCES:
            log.info(f"Downloading {source['year']} data: {source['url']}")
            try:
                content = download_with_retry(source["url"])
                df = parse_efsa_xlsx(content, source["year"])
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                log.error(f"Failed to fetch {source['year']} data: {e}")

    if not all_frames:
        log.error("No data extracted.")
        sys.exit(1)

    efsa_df = pd.concat(all_frames, ignore_index=True)

    log.info(
        f"\nExtracted EFSA data: {len(efsa_df):,} rows | "
        f"{efsa_df['organism'].nunique()} organisms | "
        f"{efsa_df['country'].nunique()} countries"
    )
    log.info(f"Source types: {efsa_df['source_type'].value_counts().to_dict()}")
    print("\nPreview:")
    print(efsa_df.head(10).to_string())
    print()

    if args.dry_run:
        log.info("--dry-run: not writing.")
        sys.exit(0)

    # ── Merge with existing CSV ───────────────────────────────────────
    # Remove source_type column before merging (not in existing schema)
    # but keep it as metadata — add to existing CSV if not present
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        log.info(f"Existing rows: {len(existing):,}")

        if "source" not in existing.columns:
            existing["source"] = "ORIGINAL"
        if "source_type" not in existing.columns:
            existing["source_type"] = "human"

        # Remove old EFSA rows to avoid duplicates on re-run
        existing = existing[existing["source"] != "EFSA"]

        combined = pd.concat([existing, efsa_df], ignore_index=True)
    else:
        combined = efsa_df

    combined = combined.drop_duplicates(
        subset=["country", "year", "organism", "antibiotic",
                "pct_resistant", "source_type"]
    ).sort_values(
        ["organism", "antibiotic", "country", "year"]
    ).reset_index(drop=True)

    log.info(f"Combined rows: {len(combined):,}")

    # ── Backup + write ────────────────────────────────────────────────
    if OUTPUT_CSV.exists():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"amr_tidy_{ts}.csv"
        backup.write_bytes(OUTPUT_CSV.read_bytes())
        log.info(f"Backup → {backup.name}")

    combined.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Written → {OUTPUT_CSV}")

    # ── Reload SQLite ─────────────────────────────────────────────────
    reload_sqlite(OUTPUT_CSV)

    # ── Save state ────────────────────────────────────────────────────
    state = {
        "last_fetch":     datetime.now().isoformat(),
        "efsa_rows":      len(efsa_df),
        "total_rows":     len(combined),
        "organisms":      combined["organism"].nunique(),
        "countries":      combined["country"].nunique(),
        "source_types":   efsa_df["source_type"].value_counts().to_dict(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    log.info("=" * 60)
    log.info(f"  Done. Database now has {len(combined):,} total rows.")
    log.info(
        f"  Human (ECDC): {len(combined[combined['source'] != 'EFSA']):,} rows"
    )
    log.info(f"  Animal/food (EFSA): {len(efsa_df):,} rows")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
