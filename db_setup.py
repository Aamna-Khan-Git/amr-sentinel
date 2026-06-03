"""
db_setup.py
-----------
Loads amr_tidy.csv into SQLite and provides structured query helpers.
Database path is read from config (DATABASE_PATH env var).
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional

from config import DATABASE_PATH

DB_PATH  = DATABASE_PATH
CSV_PATH = "data/amr_tidy.csv"

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS amr_data (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    country           TEXT    NOT NULL,
    year              INTEGER NOT NULL,
    organism          TEXT    NOT NULL,
    antibiotic        TEXT    NOT NULL,
    pct_resistant     REAL,
    total_isolates    INTEGER,
    source            TEXT,
    source_type       TEXT
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_country    ON amr_data (country);",
    "CREATE INDEX IF NOT EXISTS idx_organism   ON amr_data (organism);",
    "CREATE INDEX IF NOT EXISTS idx_antibiotic ON amr_data (antibiotic);",
    "CREATE INDEX IF NOT EXISTS idx_year       ON amr_data (year);",
    "CREATE INDEX IF NOT EXISTS idx_source_type ON amr_data (source_type);",
]

# Column name aliases — maps CSV columns to DB columns
COLUMN_MAP = {
    "percent_resistant": "pct_resistant",
    "n_isolates":        "total_isolates",
    "matrix":            "source_type",
}


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ingest_csv(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> int:
    """
    Read csv_path and reload the amr_data table.
    Called automatically by fetch_efsa.py and pdf_ingest.py after data updates.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise column names
    df = df.rename(columns=COLUMN_MAP)

    # Ensure required columns exist with defaults
    for col, default in [
        ("pct_resistant",  None),
        ("total_isolates", 0),
        ("source",         "ORIGINAL"),
        ("source_type",    "human"),
    ]:
        if col not in df.columns:
            df[col] = default

    # Keep only DB columns
    db_cols = ["country", "year", "organism", "antibiotic",
               "pct_resistant", "total_isolates", "source", "source_type"]
    df = df[[c for c in db_cols if c in df.columns]]

    conn = get_connection(db_path)
    with conn:
        conn.execute("DROP TABLE IF EXISTS amr_data;")
        conn.execute(CREATE_TABLE_SQL)
        for sql in CREATE_INDEX_SQL:
            conn.execute(sql)
        df.to_sql("amr_data", conn, if_exists="append", index=False)

    rows = len(df)
    print(f"[db_setup] Loaded {rows} rows into '{db_path}'.")
    return rows


# Keep old name as alias for backwards compatibility
load_csv_to_db = ingest_csv


# ── Query helpers ─────────────────────────────────────────────────────────────

def query_trend(
    organism: str,
    antibiotic: str,
    country: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    conn = get_connection(db_path)
    if country:
        sql = """
            SELECT year,
                   country,
                   ROUND(AVG(pct_resistant), 2) AS avg_pct_resistant,
                   SUM(total_isolates)          AS total_isolates
            FROM amr_data
            WHERE LOWER(organism)   = LOWER(:organism)
              AND LOWER(antibiotic) = LOWER(:antibiotic)
              AND LOWER(country)    = LOWER(:country)
            GROUP BY year, country
            ORDER BY year;
        """
        params = {"organism": organism, "antibiotic": antibiotic, "country": country}
    else:
        sql = """
            SELECT year,
                   ROUND(AVG(pct_resistant), 2) AS avg_pct_resistant,
                   SUM(total_isolates)          AS total_isolates,
                   COUNT(DISTINCT country)      AS n_countries
            FROM amr_data
            WHERE LOWER(organism)   = LOWER(:organism)
              AND LOWER(antibiotic) = LOWER(:antibiotic)
            GROUP BY year
            ORDER BY year;
        """
        params = {"organism": organism, "antibiotic": antibiotic}

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_country_comparison(
    organism: str,
    antibiotic: str,
    year: Optional[int] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    conn        = get_connection(db_path)
    year_filter = "AND year = :year" if year else ""
    sql = f"""
        SELECT country,
               ROUND(AVG(pct_resistant), 2) AS avg_pct_resistant,
               SUM(total_isolates)          AS total_isolates,
               MIN(year)                    AS first_year,
               MAX(year)                    AS last_year
        FROM amr_data
        WHERE LOWER(organism)   = LOWER(:organism)
          AND LOWER(antibiotic) = LOWER(:antibiotic)
          AND country != 'EU/EEA'
          {year_filter}
        GROUP BY country
        ORDER BY avg_pct_resistant DESC;
    """
    params: dict = {"organism": organism, "antibiotic": antibiotic}
    if year:
        params["year"] = year

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_top_resistant_pairs(
    country: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
    db_path: str = DB_PATH,
) -> list[dict]:
    conn    = get_connection(db_path)
    filters = []
    params: dict = {}
    if country:
        filters.append("LOWER(country) = LOWER(:country)")
        params["country"] = country
    if year:
        filters.append("year = :year")
        params["year"] = year

    filters.append("country != 'EU/EEA'")
    where_clause = ("WHERE " + " AND ".join(filters)) if filters else "WHERE country != 'EU/EEA'"
    sql = f"""
        SELECT organism,
               antibiotic,
               ROUND(AVG(pct_resistant), 2) AS avg_pct_resistant,
               SUM(total_isolates)          AS total_isolates,
               COUNT(DISTINCT country)      AS n_countries
        FROM amr_data
        {where_clause}
        GROUP BY organism, antibiotic
        ORDER BY avg_pct_resistant DESC
        LIMIT :limit;
    """
    params["limit"] = limit

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_distinct_values(field: str, db_path: str = DB_PATH) -> list[str]:
    allowed = {"country", "organism", "antibiotic", "year", "source_type"}
    if field not in allowed:
        raise ValueError(f"field must be one of {allowed}")
    conn = get_connection(db_path)
    rows = conn.execute(
        f"SELECT DISTINCT {field} FROM amr_data ORDER BY {field};"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_csv(replace=True)
