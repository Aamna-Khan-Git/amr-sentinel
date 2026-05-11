"""
db_setup.py
-----------
Loads amr_tidy.csv into a SQLite database and provides structured query helpers
for trend analysis and country/organism comparisons.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = "amr_sentinel.db"
CSV_PATH = "amr_outputs/amr_tidy.csv"


# ---------------------------------------------------------------------------
# Schema & Ingestion
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS amr_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    country         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    matrix          TEXT,
    organism        TEXT    NOT NULL,
    antibiotic      TEXT    NOT NULL,
    percent_resistant REAL,
    n_isolates      INTEGER
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_country   ON amr_data (country);",
    "CREATE INDEX IF NOT EXISTS idx_organism  ON amr_data (organism);",
    "CREATE INDEX IF NOT EXISTS idx_antibiotic ON amr_data (antibiotic);",
    "CREATE INDEX IF NOT EXISTS idx_year      ON amr_data (year);",
]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set to dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_csv_to_db(
    csv_path: str = CSV_PATH,
    db_path: str = DB_PATH,
    replace: bool = False,
) -> int:
    """
    Read *csv_path* and insert rows into the SQLite ``amr_data`` table.

    Parameters
    ----------
    csv_path : str
        Path to the tidy AMR CSV file.
    db_path : str
        Path for the SQLite database file.
    replace : bool
        If True, drop and recreate the table before loading.

    Returns
    -------
    int
        Number of rows inserted.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "country", "year", "matrix", "organism",
        "antibiotic", "percent_resistant", "n_isolates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    conn = get_connection(db_path)
    with conn:
        if replace:
            conn.execute("DROP TABLE IF EXISTS amr_data;")
        conn.execute(CREATE_TABLE_SQL)
        for sql in CREATE_INDEX_SQL:
            conn.execute(sql)

        if_exists = "replace" if replace else "append"
        df.to_sql("amr_data", conn, if_exists=if_exists, index=False)

    rows = len(df)
    print(f"[db_setup] Loaded {rows} rows into '{db_path}'.")
    return rows


# ---------------------------------------------------------------------------
# Query Helpers
# ---------------------------------------------------------------------------

def query_trend(
    organism: str,
    antibiotic: str,
    country: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Return year-by-year resistance trend for a given organism/antibiotic pair.

    Aggregates across countries when *country* is None; filters to a single
    country otherwise. Returns rows sorted by year.
    """
    conn = get_connection(db_path)
    if country:
        sql = """
            SELECT year,
                   country,
                   ROUND(AVG(percent_resistant), 2) AS avg_pct_resistant,
                   SUM(n_isolates)                  AS total_isolates
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
                   ROUND(AVG(percent_resistant), 2) AS avg_pct_resistant,
                   SUM(n_isolates)                  AS total_isolates,
                   COUNT(DISTINCT country)          AS n_countries
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
    """
    Compare resistance levels across European countries for a given
    organism/antibiotic combination.

    If *year* is supplied, restrict to that year; otherwise aggregate all years.
    """
    conn = get_connection(db_path)
    year_filter = "AND year = :year" if year else ""
    sql = f"""
        SELECT country,
               ROUND(AVG(percent_resistant), 2) AS avg_pct_resistant,
               SUM(n_isolates)                  AS total_isolates,
               MIN(year)                        AS first_year,
               MAX(year)                        AS last_year
        FROM amr_data
        WHERE LOWER(organism)   = LOWER(:organism)
          AND LOWER(antibiotic) = LOWER(:antibiotic)
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
    """
    Return the organism/antibiotic pairs with the highest average resistance.

    Optionally filter by country and/or year.
    """
    conn = get_connection(db_path)
    filters = []
    params: dict = {}
    if country:
        filters.append("LOWER(country) = LOWER(:country)")
        params["country"] = country
    if year:
        filters.append("year = :year")
        params["year"] = year

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""
    sql = f"""
        SELECT organism,
               antibiotic,
               ROUND(AVG(percent_resistant), 2) AS avg_pct_resistant,
               SUM(n_isolates)                  AS total_isolates,
               COUNT(DISTINCT country)          AS n_countries
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
    """
    Return sorted distinct values for a given column name in ``amr_data``.

    Allowed fields: country, organism, antibiotic, year, matrix.
    """
    allowed = {"country", "organism", "antibiotic", "year", "matrix"}
    if field not in allowed:
        raise ValueError(f"field must be one of {allowed}")
    conn = get_connection(db_path)
    rows = conn.execute(f"SELECT DISTINCT {field} FROM amr_data ORDER BY {field};").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_csv_to_db(replace=True)
