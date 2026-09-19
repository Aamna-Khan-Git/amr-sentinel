"""
fetch_efsa.py
-------------
Fetches and ingests EU One Health AMR Zoonoses data from Zenodo into
amr_sentinel.db.

Sources per release:
- Annex C: Indicator E. coli (pigs, cattle, broilers, turkeys, meat)
- Annex A.2: Salmonella in food-producing animals

Multi-year ingest: each Zenodo release covers one or more reporting years
under a single record (doi/record id). To add a year, add an entry to
RELEASES below with that year's Zenodo record id and reporting year, then
run with --year <YEAR> or --all-years. Re-running a year only replaces
that year's EFSA rows (DELETE is scoped to source=EFSA_ECDC_<YEAR>), so
prior years already in the DB are untouched.

Usage:
    python3 fetch_efsa.py --year 2023
    python3 fetch_efsa.py --all-years
"""
import os, sys, logging, argparse, requests, sqlite3
from pathlib import Path
from io import BytesIO
import openpyxl
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DB_PATH   = os.getenv("DATABASE_PATH", "amr_sentinel.db")
DATA_DIR  = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── Known releases ────────────────────────────────────────────────────
# One entry per EFSA-ECDC EU Summary Report. Add prior years here as you
# source their Zenodo record ids — each is a separate DOI/record, not a
# parameter on today's one. The 2022-2023 entry below is the release this
# script originally shipped with; verify the file naming is unchanged
# before trusting a newly-added year without a manual check.
RELEASES = {
    2023: {
        "zenodo_base": "https://zenodo.org/records/14645440/files",
        "ecoli":      "Annex%20C_Indicator%20E.%20coli_EFSA-ECDC_EUSR_AMR_2022-2023.xlsm?download=1",
        "salmonella": "Annex%20A.2_Salmonella_food_producing_animals_EFSA-ECDC_EUSR_AMR_2022-2023.xlsm?download=1",
    },
    # 2022: {
    #     "zenodo_base": "https://zenodo.org/records/<record_id>/files",
    #     "ecoli":       "<filename>.xlsm?download=1",
    #     "salmonella":  "<filename>.xlsm?download=1",
    # },
}

# Map sheet names to source_type labels
ECOLI_SHEETS = {
    "T. 1. Pigs E. coli":           "animal_pig",
    "T. 3. Broilers E. coli":       "animal_broiler",
    "T. 5. Pig meat BCP E. coli":   "meat_pork",
    "T. 7. Broiler meat BCP E. coli": "meat_broiler",
}

SALMONELLA_SHEETS = {
    "Pigs Salmonella spp.":    ("Salmonella spp.", "animal_pig"),
    "Pigs S. Derby":           ("S. Derby",        "animal_pig"),
    "Pigs S. Typhimurium":     ("S. Typhimurium",  "animal_pig"),
    "Pigs S. monophasic":      ("S. Typhimurium (mono)", "animal_pig"),
    "Broilers Salmonella spp.":("Salmonella spp.", "animal_broiler"),
    "Broilers S. Infantis":    ("S. Infantis",     "animal_broiler"),
    "Broilers S. Kentucky":    ("S. Kentucky",     "animal_broiler"),
    "Broilers S. Enteritidis": ("S. Enteritidis",  "animal_broiler"),
}

ANTIBIOTICS = ["GEN","AMK","CHL","AMP","CTX","CAZ","MEM","TGC",
            "NAL","CIP","AZM","COL","SMX","TMP","TET"]


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        log.info("Using cached %s", dest)
        return dest
    log.info("Downloading %s ...", dest.name)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("Saved %s (%.1f KB)", dest.name, len(r.content)/1024)
    return dest


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS amr_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT, year INTEGER, organism TEXT,
            antibiotic TEXT, pct_resistant REAL,
            total_isolates INTEGER, source TEXT, source_type TEXT
        )""")
    conn.commit()
    return conn


def parse_pct(val):
    """Convert percentage value to float, handling strings like '1.7'."""
    if val is None:
        return None
    try:
        return float(str(val).replace('%','').strip())
    except:
        return None


def ingest_ecoli(conn, path: Path, year: int):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    inserted = 0
    for sheet_name, source_type in ECOLI_SHEETS.items():
        if sheet_name not in wb.sheetnames:
            log.warning("Sheet not found: %s", sheet_name)
            continue
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = rows[0]
        # Build antibiotic column index map
        ab_idx = {}
        for ab in ANTIBIOTICS:
            for i, h in enumerate(header):
                if h and ab in str(h).upper().replace(' ',''):
                    ab_idx[ab] = i
                    break
        n_idx = next((i for i, h in enumerate(header) if h == 'N'), None)

        for row in rows[1:]:
            country = row[0]
            if not country or str(country).strip() in ('', 'EU/EEA', 'Total'):
                continue
            n_isolates = int(row[n_idx]) if n_idx and row[n_idx] else 0
            for ab, idx in ab_idx.items():
                pct = parse_pct(row[idx])
                if pct is None:
                    continue
                conn.execute("""
                    INSERT INTO amr_data
                    (country, year, organism, antibiotic, pct_resistant,
                    total_isolates, source, source_type)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (str(country).strip(), year, "E. coli", ab,
                    pct, n_isolates, f"EFSA_ECDC_{year}", source_type))
                inserted += 1
    conn.commit()
    log.info("E. coli: inserted %d rows", inserted)


def ingest_salmonella(conn, path: Path, year: int):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    inserted = 0
    for sheet_name, (organism, source_type) in SALMONELLA_SHEETS.items():
        if sheet_name not in wb.sheetnames:
            log.warning("Sheet not found: %s", sheet_name)
            continue
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = rows[0]
        ab_idx = {}
        for ab in ANTIBIOTICS:
            for i, h in enumerate(header):
                if h and ab in str(h).upper().replace(' ',''):
                    ab_idx[ab] = i
                    break
        n_idx = next((i for i, h in enumerate(header) if h == 'N'), None)

        for row in rows[1:]:
            country = row[0]
            if not country or str(country).strip() in ('', 'EU/EEA', 'Total'):
                continue
            n_isolates = int(row[n_idx]) if n_idx and row[n_idx] else 0
            for ab, idx in ab_idx.items():
                pct = parse_pct(row[idx])
                if pct is None:
                    continue
                conn.execute("""
                    INSERT INTO amr_data
                    (country, year, organism, antibiotic, pct_resistant,
                     total_isolates, source, source_type)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (str(country).strip(), year, organism, ab,
                      pct, n_isolates, f"EFSA_ECDC_{year}", source_type))
                inserted += 1
    conn.commit()
    log.info("Salmonella: inserted %d rows", inserted)


def ingest_year(conn, year: int):
    if year not in RELEASES:
        log.error("No release registered for year %d. Add it to RELEASES.", year)
        return False

    release = RELEASES[year]
    base    = release["zenodo_base"]

    # Scoped delete: only this year's EFSA rows, so re-running one year
    # never touches other years already in the DB.
    source = f"EFSA_ECDC_{year}"
    deleted = conn.execute(
        "DELETE FROM amr_data WHERE source = ?", (source,)
    ).rowcount
    conn.commit()
    log.info("Cleared %d existing rows for %s (idempotent re-run)", deleted, source)

    ecoli_path = download_file(
        f"{base}/{release['ecoli']}", DATA_DIR / f"efsa_ecoli_{year}.xlsm"
    )
    salmonella_path = download_file(
        f"{base}/{release['salmonella']}", DATA_DIR / f"efsa_salmonella_{year}.xlsm"
    )

    ingest_ecoli(conn, ecoli_path, year)
    ingest_salmonella(conn, salmonella_path, year)

    total = conn.execute(
        "SELECT COUNT(*) FROM amr_data WHERE source = ?", (source,)
    ).fetchone()[0]
    log.info("Total %s rows in DB: %d", source, total)
    return True


def main():
    parser = argparse.ArgumentParser(description="EFSA-ECDC AMR data fetcher")
    parser.add_argument("--year", type=int, help="Ingest a single reporting year")
    parser.add_argument("--all-years", action="store_true",
                        help="Ingest every year registered in RELEASES")
    args = parser.parse_args()

    conn = get_connection()

    if args.all_years:
        years = sorted(RELEASES)
    elif args.year:
        years = [args.year]
    else:
        # Default: only the most recent registered year, to match the
        # script's old single-year behaviour when run with no flags.
        years = [max(RELEASES)]
        log.info("No --year given — defaulting to latest registered year %d. "
                "Use --all-years to (re)ingest every registered year.", years[0])

    for year in years:
        ingest_year(conn, year)

    overall = conn.execute(
        "SELECT MIN(year), MAX(year), COUNT(DISTINCT year) FROM amr_data "
        "WHERE source LIKE 'EFSA%'"
    ).fetchone()
    log.info("EFSA data in DB now spans %s–%s across %d year(s)", *overall)

    conn.close()


if __name__ == "__main__":
    main()