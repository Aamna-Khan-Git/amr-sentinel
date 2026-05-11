"""
AMR Pipeline – EFSA/ECDC EUSR AMR 2023-2024 Annexes
====================================================
Parses the real uploaded annex files, builds a tidy dataframe, runs
unsupervised clustering (UMAP + HDBSCAN / K-means), trains a supervised
model, and saves all outputs + plots.

Files handled
-------------
  Annex A1  – Salmonella, humans          (.xlsx, 3-row header, year in title)
  Annex A2  – Salmonella, food-prod animals (.xlsm, 1-row header, year in title)
  Annex B1  – Campylobacter, humans       (.xlsx, 3-row header)
  Annex B2  – Campylobacter, animals      (.xlsm, 1-row header)
  Annex C   – Indicator E. coli, animals  (.xlsm, 1-row header)
  Annex D1  – ESBL/AmpC/CP producers      (.xlsm, 2-row header, prevalence %)
  Annex F   – Enterococcus, animals       (.xlsx, 1-row header)

Two sheet layout types
----------------------
  TYPE_A  (3-row header):
    Row 0 – title (contains year)
    Row 1 – antibiotic names (every other column, NaN in between)
    Row 2 – "Country | N | % Res | N | % Res …"
    Data from row 3

  TYPE_B  (1-row header):
    Row 0 – "Country | N | GEN (%) | AMK (%) | …"
    Data from row 1

  TYPE_D1 (2-row header, ESBL/AmpC prevalence):
    Row 0 – "Country | Ns | ESBL and/or AmpC | …"
    Row 1 – "n | %P | 95% CI | …"
    Data from row 2  (only %P columns extracted)

Run
---
    python amr_pipeline.py

Requirements
------------
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
    pip install umap-learn hdbscan   # optional but recommended
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT  = Path("amr_outputs");  OUTPUT.mkdir(exist_ok=True)
PLOTS   = OUTPUT / "plots";     PLOTS.mkdir(exist_ok=True)


def _find_annex(keyword: str) -> str:
    """
    Search for an annex file by keyword (case-insensitive) under the script
    directory, cwd, and any subdirectory one level deep.  Returns the first
    match, or the keyword itself (triggers graceful SKIP in the parser).
    """
    kw = keyword.lower()
    roots: list[Path] = [Path(__file__).parent, Path.cwd(),
                          Path("/mnt/user-data/uploads")]
    for root in roots:
        if not root.exists():
            continue
        # Direct children + one level of subdirectories
        candidates = list(root.iterdir()) + [
            sub for d in root.iterdir()
            if d.is_dir() for sub in d.iterdir()]
        for p in candidates:
            if p.is_file() and kw in p.name.lower():
                return str(p)
    return keyword   # not found – parse_all_sheets will SKIP gracefully


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHEET CATALOGUE
#     Each entry: (filepath, sheet_name, layout_type, organism, matrix, year_override)
#     year_override=None  → extract from sheet title row
#     year_override=int   → use that year directly
# ─────────────────────────────────────────────────────────────────────────────
A1 = _find_annex("annex a.1")
A2 = _find_annex("annex a.2")
B1 = _find_annex("annex b.1")
B2 = _find_annex("annex b.2")
CC = _find_annex("annex c.")
D1 = _find_annex("annex d.1")
FF = _find_annex("annex f")

# (file, sheet, layout_type, organism, matrix, year_override)
SHEET_CATALOGUE = [
    # ── Annex A1  Salmonella humans ──────────────────────────────────────────
    (A1, "Hum Salm spp.",        "TYPE_A", "Salmonella spp.",   "human",       None),
    (A1, "Hum S. Enter",         "TYPE_A", "S. Enteritidis",    "human",       None),
    (A1, "Hum S. Typhim",        "TYPE_A", "S. Typhimurium",    "human",       None),
    (A1, "Hum mono S. Typhim",   "TYPE_A", "S. Typhimurium (mono)", "human",   None),
    (A1, "Hum S. Infan",         "TYPE_A", "S. Infantis",       "human",       None),
    (A1, "Hum S. Kenty",         "TYPE_A", "S. Kentucky",       "human",       None),
    (A1, "Hum S. Derby",         "TYPE_A", "S. Derby",          "human",       None),

    # ── Annex A2  Salmonella animals ─────────────────────────────────────────
    (A2, "Pigs Salmonella spp.",    "TYPE_B", "Salmonella spp.", "pigs",        2023),
    (A2, "Pigs S. Derby",          "TYPE_B", "S. Derby",        "pigs",        2023),
    (A2, "Pigs S. Typhimurium",    "TYPE_B", "S. Typhimurium",  "pigs",        2023),
    (A2, "Pigs S. monophasic",     "TYPE_B", "S. Typhimurium (mono)", "pigs",  2023),
    (A2, "Cattle Salmonella spp.", "TYPE_B", "Salmonella spp.", "cattle",      2023),
    (A2, "Cattle S. Derby",        "TYPE_B", "S. Derby",        "cattle",      2023),
    (A2, "Cattle S. Typhimurium",  "TYPE_B", "S. Typhimurium",  "cattle",      2023),
    (A2, "Cattle S. monophasic",   "TYPE_B", "S. Typhimurium (mono)", "cattle", 2023),
    (A2, "Broilers Salmonella spp.","TYPE_B","Salmonella spp.", "broilers",    2024),
    (A2, "Broilers S. Infantis",   "TYPE_B", "S. Infantis",     "broilers",    2024),
    (A2, "Broilers S. Kentucky",   "TYPE_B", "S. Kentucky",     "broilers",    2024),
    (A2, "Broilers S. Enteritidis","TYPE_B", "S. Enteritidis",  "broilers",    2024),
    (A2, "Laying hens Salmonella spp.","TYPE_B","Salmonella spp.","laying_hens",2024),
    (A2, "Laying hens S. Infantis","TYPE_B", "S. Infantis",     "laying_hens", 2024),
    (A2, "Laying hens S. Kentucky","TYPE_B", "S. Kentucky",     "laying_hens", 2024),
    (A2, "Laying hens S. Enteritidis","TYPE_B","S. Enteritidis","laying_hens", 2024),
    (A2, "Turkeys Salmonella spp.","TYPE_B", "Salmonella spp.", "turkeys",     2024),
    (A2, "Turkeys S. Infantis",    "TYPE_B", "S. Infantis",     "turkeys",     2024),
    (A2, "Turkeys S. Kentucky",    "TYPE_B", "S. Kentucky",     "turkeys",     2024),
    (A2, "Turkeys S. Enteritidis", "TYPE_B", "S. Enteritidis",  "turkeys",     2024),

    # ── Annex B1  Campylobacter humans ───────────────────────────────────────
    (B1, "Hum C. jejuni",          "TYPE_A", "C. jejuni",       "human",       None),
    (B1, "Hum C. coli",            "TYPE_A", "C. coli",         "human",       None),

    # ── Annex B2  Campylobacter animals ──────────────────────────────────────
    (B2, "Broilers C. coli",       "TYPE_B", "C. coli",         "broilers",    2024),
    (B2, "Broilers C. jejuni",     "TYPE_B", "C. jejuni",       "broilers",    2024),
    (B2, "Turkeys C. coli",        "TYPE_B", "C. coli",         "turkeys",     2024),
    (B2, "Turkeys C. jejuni",      "TYPE_B", "C. jejuni",       "turkeys",     2024),
    (B2, "Pigs C. coli",           "TYPE_B", "C. coli",         "pigs",        2023),
    (B2, "Pigs C. jejuni",         "TYPE_B", "C. jejuni",       "pigs",        2023),
    (B2, "Cattle C. coli",         "TYPE_B", "C. coli",         "cattle",      2023),
    (B2, "Cattle C. jejuni",       "TYPE_B", "C. jejuni",       "cattle",      2023),

    # ── Annex C  Indicator E. coli ───────────────────────────────────────────
    (CC, "T. 1. Pigs E. coli",     "TYPE_B", "E. coli",         "pigs",        2023),
    (CC, "T. 2. Calves E. coli",   "TYPE_B", "E. coli",         "cattle",      2023),
    (CC, "T. 3. Broilers E. coli", "TYPE_B", "E. coli",         "broilers",    2024),
    (CC, "T. 4. Turkey E. coli",   "TYPE_B", "E. coli",         "turkeys",     2024),
    (CC, "T. 5. Pig meat BCP E. coli",    "TYPE_B","E. coli",   "pig_meat",    2023),
    (CC, "T. 6. Bovine meat BCP E. coli", "TYPE_B","E. coli",   "bovine_meat", 2023),
    (CC, "T. 7. Broiler meat BCP E. coli","TYPE_B","E. coli",   "broiler_meat",2024),
    (CC, "T. 8. Turkey meat BCP E. coli", "TYPE_B","E. coli",   "turkey_meat", 2024),

    # ── Annex D1  ESBL/AmpC prevalence ───────────────────────────────────────
    (D1, "Prevalence broilers",    "TYPE_D1","E. coli (ESBL/AmpC)","broilers", 2024),
    (D1, "Prevalence turkeys",     "TYPE_D1","E. coli (ESBL/AmpC)","turkeys",  2024),
    (D1, "Prevalence pigs",        "TYPE_D1","E. coli (ESBL/AmpC)","pigs",     2023),
    (D1, "Prevalence calves",      "TYPE_D1","E. coli (ESBL/AmpC)","cattle",   2023),
    (D1, "Prevalence broiler meat","TYPE_D1","E. coli (ESBL/AmpC)","broiler_meat",2024),
    (D1, "Prevalence turkey meat", "TYPE_D1","E. coli (ESBL/AmpC)","turkey_meat",2024),
    (D1, "Prevalence pig meat",    "TYPE_D1","E. coli (ESBL/AmpC)","pig_meat", 2023),
    (D1, "Prevalence bovine meat", "TYPE_D1","E. coli (ESBL/AmpC)","bovine_meat",2023),

    # ── Annex F  Enterococcus animals ────────────────────────────────────────
    (FF, "Broilers E. faecalis",   "TYPE_B", "E. faecalis",     "broilers",    2024),
    (FF, "Broilers E. faecium",    "TYPE_B", "E. faecium",      "broilers",    2024),
    (FF, "Turkeys E. faecalis",    "TYPE_B", "E. faecalis",     "turkeys",     2024),
    (FF, "Turkeys E. faecium",     "TYPE_B", "E. faecium",      "turkeys",     2024),
    (FF, "Pigs E. faecalis",       "TYPE_B", "E. faecalis",     "pigs",        2023),
    (FF, "Pigs E. faecium",        "TYPE_B", "E. faecium",      "pigs",        2023),
    (FF, "Cattle E. faecalis",     "TYPE_B", "E. faecalis",     "cattle",      2023),
    (FF, "Cattle E. faecium",      "TYPE_B", "E. faecium",      "cattle",      2023),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HARMONISATION MAPS
# ─────────────────────────────────────────────────────────────────────────────
# Antibiotic column name → harmonised code
AB_MAP = {
    # full names (in TYPE_A row-1 headers)
    "gentamicin":                  "GEN",
    "amikacin":                    "AMK",
    "chloramphenicol":             "CHL",
    "ampicillin":                  "AMP",
    "cefotaxime":                  "CTX",
    "ceftazidime":                 "CAZ",
    "meropenem":                   "MEM",
    "tigecycline":                 "TGC",
    "nalidixic acid":              "NAL",
    "ciprofloxacin":               "CIP",
    "ciprofloxacin/ pefloxacin":   "CIP",
    "ciprofloxacin/pefloxacin":    "CIP",
    "erythromycin":                "ERY",
    "azithromycin":                "AZM",
    "colistin":                    "COL",
    "sulfamethoxazole":            "SMX",
    "trimethoprim":                "TMP",
    "co-trimoxazole":              "SXT",
    "tetracycline":                "TET",
    "co-amoxiclav":                "AMC",
    "ertapenem":                   "ETP",
    "quinupristin/dalfopristin":   "QDA",
    "linezolid":                   "LZD",
    "vancomycin":                  "VAN",
    "teicoplanin":                 "TEC",
    "daptomycin":                  "DPT",
    # abbreviated (in TYPE_B column headers like "GEN (%)")
    "gen":  "GEN", "amk": "AMK", "chl": "CHL", "amp": "AMP",
    "ctx":  "CTX", "caz": "CAZ", "mem": "MEM", "tgc": "TGC",
    "nal":  "NAL", "cip": "CIP", "ery": "ERY", "azm": "AZM",
    "col":  "COL", "smx": "SMX", "tmp": "TMP", "tet": "TET",
    "etp":  "ETP", "q/d": "QDA", "lzd": "LZD", "van": "VAN",
    "tec":  "TEC", "dpt": "DPT", "amc": "AMC",
}

AB_CLASS = {
    "GEN":"Aminoglycosides","AMK":"Aminoglycosides",
    "CHL":"Phenicols",
    "AMP":"Penicillins","AMC":"Penicillins",
    "CTX":"Cephalosporins","CAZ":"Cephalosporins",
    "MEM":"Carbapenems","ETP":"Carbapenems",
    "TGC":"Tetracyclines","TET":"Tetracyclines",
    "NAL":"Quinolones","CIP":"Fluoroquinolones",
    "ERY":"Macrolides","AZM":"Macrolides",
    "COL":"Polymyxins",
    "SMX":"Sulfonamides","SXT":"Sulfonamides","TMP":"Trimethoprim",
    "QDA":"Streptogramins","LZD":"Oxazolidinones",
    "VAN":"Glycopeptides","TEC":"Glycopeptides",
    "DPT":"Lipopeptides",
}

# Country name cleanup
COUNTRY_FIX = {
    "total mss":              None,  # aggregate – drop
    "total non-mss":          None,
    "total mss and non-mss":  None,
    "median mss and non-mss": None,
    "median":                 None,
    "uk (northern ireland)":  "United Kingdom (NI)",
    "united kingdom (northern ireland)": "United Kingdom (NI)",
    "republic of north macedonia": "North Macedonia",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
}


def _clean_country(raw: Any) -> str | None:
    """Normalise country name; return None if it's an aggregate/footnote row."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    # Remove footnote markers like (a), (b), *, †
    s = re.sub(r"\s*[\(\*†‡§]+[a-z,\d]*[\)\s]*$", "", s, flags=re.I).strip()
    key = s.lower()
    if key in COUNTRY_FIX:
        return COUNTRY_FIX[key]
    # Drop if it starts with "N:", "(", or is a footnote
    if s.startswith(("N:", "(", "Note", "Source", "–", "-", "a)", "b)")):
        return None
    if len(s) < 2:
        return None
    return s.title()


def _to_float(v: Any) -> float:
    """Convert a cell value to float %R; return NaN on failure."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(",", ".")
    if s in ("-", "–", "n/a", "na", "nd", "nr", ""):
        return np.nan
    s = re.sub(r"[<>≤≥]", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _year_from_title(filepath: str, sheet_name: str) -> int | None:
    """Read row 0 of a sheet and extract the last 4-digit year found."""
    try:
        raw = pd.read_excel(filepath, sheet_name=sheet_name,
                            header=None, nrows=1, engine="openpyxl")
        title = str(raw.iloc[0, 0])
        years = re.findall(r"\b(20[12]\d)\b", title)
        return int(years[-1]) if years else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_type_a(filepath: str, sheet_name: str,
                  organism: str, matrix: str, year_override: int | None) -> pd.DataFrame:
    """
    TYPE_A: 3-row header (title / antibiotic names / Country,N,%Res…)
    Antibiotic names live in row 1 (every other column starting at col 1).
    Data starts at row 3.
    """
    year = year_override or _year_from_title(filepath, sheet_name)

    raw = pd.read_excel(filepath, sheet_name=sheet_name,
                        header=None, engine="openpyxl")

    # Build antibiotic label for every column pair (N, %Res)
    ab_row = raw.iloc[1]  # antibiotics row
    hdr_row = raw.iloc[2]  # Country / N / %Res row

    # Map column index → antibiotic code  (ab name appears at col i, %Res at col i+2)
    col_ab: dict[int, str] = {}
    current_ab = None
    for i, val in enumerate(ab_row):
        if pd.notna(val):
            current_ab = str(val).strip().lower()
            current_ab = re.sub(r"\s*\(.*?\)", "", current_ab).strip()
            current_ab = re.sub(r"[^a-z/ -]", "", current_ab).strip()
        if pd.notna(hdr_row.iloc[i]) and str(hdr_row.iloc[i]).strip() == "% Res":
            code = AB_MAP.get(current_ab)
            if code and i not in col_ab:
                col_ab[i] = code

    records = []
    for _, row in raw.iloc[3:].iterrows():
        country = _clean_country(row.iloc[0])
        if country is None:
            continue
        for col_idx, ab_code in col_ab.items():
            pct = _to_float(row.iloc[col_idx])
            if np.isnan(pct):
                continue
            # n_isolates is in the column before %Res
            n = _to_float(row.iloc[col_idx - 1])
            records.append({
                "country": country, "year": year,
                "matrix": matrix, "organism": organism,
                "antibiotic": ab_code, "percent_resistant": pct,
                "n_isolates": n,
            })

    return pd.DataFrame(records)


def _parse_type_b(filepath: str, sheet_name: str,
                  organism: str, matrix: str, year_override: int | None) -> pd.DataFrame:
    """
    TYPE_B: 1-row header  Country | N | GEN (%) | AMK (%) | …
    Data starts at row 1.
    """
    year = year_override or _year_from_title(filepath, sheet_name)

    df = pd.read_excel(filepath, sheet_name=sheet_name,
                       header=0, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    # Find country col (first col)
    country_col = df.columns[0]

    # Find N col
    n_col = next((c for c in df.columns if re.match(r"^N$", c, re.I)), None)

    # Find antibiotic %  columns: pattern "XXX (%)" or "XXX(%)"
    ab_cols: dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"^([A-Za-z/]+)\s*\(%\)$", col.strip())
        if m:
            raw_ab = m.group(1).strip().lower()
            code = AB_MAP.get(raw_ab)
            if code:
                ab_cols[col] = code

    records = []
    for _, row in df.iterrows():
        country = _clean_country(row[country_col])
        if country is None:
            continue
        n = _to_float(row[n_col]) if n_col else np.nan
        for col, ab_code in ab_cols.items():
            pct = _to_float(row[col])
            if np.isnan(pct):
                continue
            records.append({
                "country": country, "year": year,
                "matrix": matrix, "organism": organism,
                "antibiotic": ab_code, "percent_resistant": pct,
                "n_isolates": n,
            })

    return pd.DataFrame(records)


def _parse_type_d1(filepath: str, sheet_name: str,
                   organism: str, matrix: str, year_override: int | None) -> pd.DataFrame:
    """
    TYPE_D1: 2-row header (phenotype names / n, %P, 95% CI, …)
    Extracts the ESBL and/or AmpC %P column as the key resistance indicator.
    """
    year = year_override

    raw = pd.read_excel(filepath, sheet_name=sheet_name,
                        header=None, engine="openpyxl")

    # Row 0 = phenotype group headers, Row 1 = n / %P / CI sub-headers
    grp_row = raw.iloc[0]
    sub_row = raw.iloc[1]

    # Build column map: find %P columns and their parent group
    col_map: dict[int, str] = {}
    current_grp = None
    for i, val in enumerate(grp_row):
        if pd.notna(val) and str(val).strip():
            current_grp = str(val).strip()
        if (pd.notna(sub_row.iloc[i]) and
                str(sub_row.iloc[i]).strip().lower() in ("%p", "% p", "%")):
            # Only keep "ESBL and/or AmpC" and pure "ESBL" %P columns
            if current_grp and ("esbl" in current_grp.lower() or
                                 "ampc" in current_grp.lower()):
                col_map[i] = current_grp

    records = []
    for _, row in raw.iloc[2:].iterrows():
        country = _clean_country(row.iloc[0])
        if country is None:
            continue
        for col_idx, grp_name in col_map.items():
            pct = _to_float(row.iloc[col_idx])
            if np.isnan(pct):
                continue
            # Assign pseudo antibiotic code based on phenotype
            if "cp" in grp_name.lower() or "carbapenem" in grp_name.lower():
                ab_code = "CARBA_PHENO"
            elif "esbl" in grp_name.lower() and "ampc" in grp_name.lower():
                ab_code = "ESBL_AmpC_PHENO"
            elif "esbl" in grp_name.lower():
                ab_code = "ESBL_PHENO"
            else:
                ab_code = "AmpC_PHENO"

            # n_isolates from col 1 (Ns)
            n = _to_float(row.iloc[1])
            records.append({
                "country": country, "year": year,
                "matrix": matrix, "organism": organism,
                "antibiotic": ab_code, "percent_resistant": pct,
                "n_isolates": n,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STEP 1 – PARSE ALL SHEETS
# ─────────────────────────────────────────────────────────────────────────────

def parse_all_sheets() -> pd.DataFrame:
    parsers = {
        "TYPE_A":  _parse_type_a,
        "TYPE_B":  _parse_type_b,
        "TYPE_D1": _parse_type_d1,
    }
    frames: list[pd.DataFrame] = []

    for (filepath, sheet_name, layout, organism, matrix, year_override) in SHEET_CATALOGUE:
        if not Path(filepath).exists():
            print(f"  [SKIP – not found] {Path(filepath).name} :: {sheet_name}")
            continue
        try:
            df = parsers[layout](filepath, sheet_name, organism, matrix, year_override)
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"  ✓ {Path(filepath).name[:35]:38s} | {sheet_name:35s} | {len(df):5d} rows")
            else:
                print(f"  [EMPTY] {sheet_name}")
        except Exception as e:
            print(f"  [ERROR] {sheet_name}: {e}")

    if not frames:
        raise RuntimeError("No data parsed.")

    amr = pd.concat(frames, ignore_index=True)

    # De-duplicate (average %R when same key appears twice)
    key = ["country", "year", "matrix", "organism", "antibiotic"]
    amr = (amr.groupby(key, as_index=False)
              .agg(percent_resistant=("percent_resistant", "mean"),
                   n_isolates=("n_isolates", "mean")))

    amr["antibiotic_class"] = amr["antibiotic"].map(AB_CLASS).fillna("Other")

    # Broad matrix grouping for supervised model
    amr["matrix_group"] = amr["matrix"].map({
        "human": "human",
        "pigs": "animal", "cattle": "animal", "broilers": "animal",
        "laying_hens": "animal", "turkeys": "animal",
        "pig_meat": "meat", "bovine_meat": "meat",
        "broiler_meat": "meat", "turkey_meat": "meat",
    }).fillna("animal")

    print(f"\n  ✓ Tidy AMR table: {len(amr):,} rows | "
          f"{amr['country'].nunique()} countries | "
          f"{amr['organism'].nunique()} organisms | "
          f"{amr['antibiotic'].nunique()} antibiotics\n")
    return amr


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STEP 2 – CLUSTERING  (UMAP + HDBSCAN / K-means)
# ─────────────────────────────────────────────────────────────────────────────

def build_country_features(amr: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per (country, matrix_group, organism) with
    one column per antibiotic.  Used as the feature matrix for clustering.
    """
    # Keep only rows with known year
    df = amr.dropna(subset=["year", "percent_resistant"])

    # For each (country, matrix_group, organism, antibiotic) take mean over years
    grp = (df.groupby(["country", "matrix_group", "organism", "antibiotic"],
                      as_index=False)
             ["percent_resistant"].mean())

    pivot = grp.pivot_table(
        index=["country", "matrix_group", "organism"],
        columns="antibiotic",
        values="percent_resistant",
    )
    pivot.columns = [f"pct_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Summary stats per row
    feat_cols = [c for c in pivot.columns if c.startswith("pct_")]
    pivot["mean_pct"] = pivot[feat_cols].mean(axis=1)
    pivot["std_pct"]  = pivot[feat_cols].std(axis=1)
    pivot["max_pct"]  = pivot[feat_cols].max(axis=1)
    pivot["n_ab"]     = pivot[feat_cols].notna().sum(axis=1)

    return pivot


def run_clustering(feat_df: pd.DataFrame,
                   n_clusters_fallback: int = 6,
                   random_state: int = 42) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    feat_cols = [c for c in feat_df.columns
                 if c.startswith("pct_") or c in ("mean_pct", "std_pct", "max_pct")]

    X = feat_df[feat_cols].values
    imp    = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_imp  = imp.fit_transform(X)
    X_sc   = scaler.fit_transform(X_imp)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X_sc)-1),
                            random_state=random_state, min_dist=0.1)
        emb = reducer.fit_transform(X_sc)
        embed_method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        emb = PCA(2, random_state=random_state).fit_transform(X_sc)
        embed_method = "PCA"

    feat_df = feat_df.copy()
    feat_df["embed_x"] = emb[:, 0]
    feat_df["embed_y"] = emb[:, 1]

    # Clustering
    try:
        import hdbscan
        cl  = hdbscan.HDBSCAN(min_cluster_size=max(3, len(X_sc)//20),
                               gen_min_span_tree=True)
        labels = cl.fit_predict(emb)
        cl_method = "HDBSCAN"
    except ImportError:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters_fallback,
                        random_state=random_state, n_init=10).fit_predict(X_sc)
        cl_method = "K-means"

    feat_df["cluster_id"]     = labels
    feat_df["cluster_method"] = cl_method
    feat_df["embed_method"]   = embed_method

    n_cl = len(set(labels) - {-1})
    print(f"  ✓ {embed_method} + {cl_method}: {n_cl} clusters "
          f"(noise: {(labels==-1).sum()})\n")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  STEP 3 – SUPERVISED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_supervised_dataset(amr: pd.DataFrame) -> pd.DataFrame:
    """
    Predict human %R for a given antibiotic from animal/meat %R (same year)
    and summary statistics of the animal/meat resistance profile.
    """
    human  = amr[amr["matrix_group"] == "human"].copy()
    animal = amr[amr["matrix_group"] != "human"].copy()

    # Mean animal %R by (country, year, organism, antibiotic)
    an_mean = (animal.groupby(["country", "year", "organism", "antibiotic"],
                               as_index=False)
                     ["percent_resistant"].mean()
                     .rename(columns={"percent_resistant": "animal_pct_R"}))

    merged = pd.merge(human, an_mean,
                      on=["country", "year", "organism", "antibiotic"],
                      how="inner")

    # Overall animal resistance burden: mean across all antibiotics
    burden = (animal.groupby(["country", "year", "matrix_group"],
                              as_index=False)
                    ["percent_resistant"].mean()
                    .rename(columns={"percent_resistant": "animal_mean_burden",
                                     "matrix_group": "matrix_group_burden"}))
    # Collapse over matrix_group
    burden = (burden.groupby(["country", "year"], as_index=False)
                    ["animal_mean_burden"].mean())

    merged = pd.merge(merged, burden, on=["country", "year"], how="left")

    # Binary high-resistance label (threshold = 25%)
    merged["high_resistance"] = (merged["percent_resistant"] >= 25).astype(int)

    return merged.dropna(subset=["percent_resistant", "animal_pct_R"])


def train_model(model_df: pd.DataFrame,
                target: str = "percent_resistant",
                test_size: float = 0.2,
                random_state: int = 42,
                n_estimators: int = 300) -> dict:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, roc_auc_score, accuracy_score)
    from sklearn.impute import SimpleImputer

    binary = target == "high_resistance"
    feature_cols = ["animal_pct_R", "animal_mean_burden"]

    df = model_df.dropna(subset=[target] + feature_cols)
    if len(df) < 20:
        print(f"  [WARN] Only {len(df)} rows – skipping supervised model.")
        return {}

    X = df[feature_cols].values
    y = df[target].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    if binary:
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     random_state=random_state, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        metrics = {
            "auroc":    float(roc_auc_score(y_te, y_proba)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
        }
        model = clf
    else:
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    random_state=random_state, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        metrics = {
            "mae":  float(mean_absolute_error(y_te, y_pred)),
            "rmse": float(mean_squared_error(y_te, y_pred) ** 0.5),
            "r2":   float(r2_score(y_te, y_pred)),
        }
        model = reg

    fi = pd.DataFrame({"feature": feature_cols,
                        "importance": model.feature_importances_})

    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")

    return {"model": model, "imputer": imp, "metrics": metrics,
            "feature_importances": fi, "feature_cols": feature_cols,
            "y_test": y_te, "y_pred": y_pred, "binary": binary}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STEP 4 – PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(amr: pd.DataFrame, feat_df: pd.DataFrame,
               model_results: dict, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)

    # ── Plot 1: Embedding scatter coloured by cluster ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    unique_cl = sorted(feat_df["cluster_id"].unique())
    cmap = plt.get_cmap("tab20", len(unique_cl))
    for i, cid in enumerate(unique_cl):
        sub = feat_df[feat_df["cluster_id"] == cid]
        lbl = f"Cluster {cid}" if cid != -1 else "Noise"
        ax.scatter(sub["embed_x"], sub["embed_y"],
                   s=25, alpha=0.75, color=cmap(i), label=lbl)
    em = feat_df["embed_method"].iloc[0]
    cl = feat_df["cluster_method"].iloc[0]
    ax.set_xlabel(f"{em}-1"); ax.set_ylabel(f"{em}-2")
    ax.set_title(f"{em} embedding – AMR resistance phenotype clusters ({cl})")
    ax.legend(loc="best", fontsize=8, markerscale=1.4, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(plots_dir / "cluster_scatter.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Mean % resistance per cluster × antibiotic (heatmap) ─────
    feat_ab_cols = [c for c in feat_df.columns if c.startswith("pct_")]
    if feat_ab_cols:
        cl_means = (feat_df[feat_df["cluster_id"] != -1]
                    .groupby("cluster_id")[feat_ab_cols].mean())
        cl_means.columns = [c.replace("pct_", "") for c in cl_means.columns]
        # Keep only antibiotics with at least some data
        cl_means = cl_means.dropna(axis=1, how="all")
        if len(cl_means) > 0 and len(cl_means.columns) > 0:
            fig, ax = plt.subplots(figsize=(max(12, len(cl_means.columns) * 0.6),
                                            max(4,  len(cl_means) * 0.5)))
            sns.heatmap(cl_means, annot=True, fmt=".1f", cmap="YlOrRd",
                        linewidths=0.3, ax=ax, cbar_kws={"label": "Mean % Resistant"})
            ax.set_title("Mean % Resistant per Cluster × Antibiotic")
            ax.set_xlabel("Antibiotic"); ax.set_ylabel("Cluster")
            fig.tight_layout()
            fig.savefig(plots_dir / "cluster_heatmap.png", dpi=150)
            plt.close(fig)
            print("  Saved: cluster_heatmap.png")

    print("  Saved: cluster_scatter.png")

    # ── Plot 3: Ciprofloxacin resistance – human vs animal, by country ────
    cip_human  = amr[(amr["antibiotic"] == "CIP") & (amr["matrix_group"] == "human")]
    cip_animal = amr[(amr["antibiotic"] == "CIP") & (amr["matrix_group"] == "animal")]
    if len(cip_human) > 0 and len(cip_animal) > 0:
        ch = cip_human.groupby("country")["percent_resistant"].mean().rename("human_CIP")
        ca = cip_animal.groupby("country")["percent_resistant"].mean().rename("animal_CIP")
        scatter = pd.merge(ch, ca, left_index=True, right_index=True).reset_index()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(scatter["animal_CIP"], scatter["human_CIP"],
                   s=40, alpha=0.7, edgecolors="none", color="steelblue")
        for _, r in scatter.iterrows():
            ax.annotate(r["country"], (r["animal_CIP"], r["human_CIP"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
        lim = max(scatter[["animal_CIP", "human_CIP"]].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Animal % Resistant (CIP, mean across matrices)")
        ax.set_ylabel("Human % Resistant (CIP, mean)")
        ax.set_title("Ciprofloxacin Resistance – Animal vs Human by Country")
        fig.tight_layout()
        fig.savefig(plots_dir / "country_CIP_scatter.png", dpi=150)
        plt.close(fig)
        print("  Saved: country_CIP_scatter.png")

    # ── Plot 4: Top antibiotics resistance profile across organisms ────────
    top_ab = (amr.groupby("antibiotic")["percent_resistant"]
                 .mean().nlargest(10).index.tolist())
    ab_org = (amr[amr["antibiotic"].isin(top_ab)]
              .groupby(["organism", "antibiotic"])["percent_resistant"].mean()
              .unstack("antibiotic"))
    if len(ab_org) > 0:
        fig, ax = plt.subplots(figsize=(12, max(4, len(ab_org) * 0.5)))
        ab_org.plot(kind="barh", ax=ax, legend=True)
        ax.set_xlabel("Mean % Resistant")
        ax.set_title("Mean % Resistance to Top 10 Antibiotics by Organism")
        ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "organism_antibiotic_profile.png", dpi=150)
        plt.close(fig)
        print("  Saved: organism_antibiotic_profile.png")

    # ── Plot 5: Feature importances ────────────────────────────────────────
    if model_results and "feature_importances" in model_results:
        fi = model_results["feature_importances"]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=fi, x="importance", y="feature", palette="viridis_r", ax=ax)
        ax.set_title("Feature Importances – Supervised Model")
        ax.set_xlabel("Mean Decrease in Impurity")
        fig.tight_layout()
        fig.savefig(plots_dir / "feature_importances.png", dpi=150)
        plt.close(fig)
        print("  Saved: feature_importances.png")

    # ── Plot 6: Predicted vs actual (regression only) ──────────────────────
    if model_results and not model_results.get("binary", True):
        y_te = model_results["y_test"]
        y_pr = model_results["y_pred"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_te, y_pr, s=18, alpha=0.4, edgecolors="none", color="teal")
        lim = max(np.nanmax(y_te), np.nanmax(y_pr)) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel("Actual % Resistant"); ax.set_ylabel("Predicted % Resistant")
        rmse = model_results["metrics"]["rmse"]
        r2   = model_results["metrics"]["r2"]
        ax.set_title(f"Predicted vs Actual  |  RMSE={rmse:.2f}  R²={r2:.3f}")
        fig.tight_layout()
        fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=150)
        plt.close(fig)
        print("  Saved: predicted_vs_actual.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  EFSA/ECDC AMR Pipeline  –  EUSR 2023-2024")
    print("=" * 70)

    # ── Step 1: Parse all annexes ────────────────────────────────────────────
    print("\n── Step 1: Parsing annex files ──────────────────────────────────────")
    amr = parse_all_sheets()
    amr.to_csv(OUTPUT / "amr_tidy.csv", index=False)
    print(f"  Saved: {OUTPUT}/amr_tidy.csv  ({len(amr):,} rows)")

    # ── Step 2: Clustering ───────────────────────────────────────────────────
    print("\n── Step 2: Feature engineering & clustering ─────────────────────────")
    feat_df = build_country_features(amr)
    feat_df = run_clustering(feat_df)

    cluster_out_cols = [c for c in feat_df.columns if not c.startswith("pct_")]
    feat_df[cluster_out_cols].to_csv(OUTPUT / "cluster_assignments.csv", index=False)
    print(f"  Saved: {OUTPUT}/cluster_assignments.csv  ({len(feat_df):,} rows)")

    # ── Step 3: Supervised model ─────────────────────────────────────────────
    print("\n── Step 3: Supervised model ─────────────────────────────────────────")
    model_df = build_supervised_dataset(amr)
    print(f"  Supervised dataset: {len(model_df):,} rows")
    model_results = {}
    if len(model_df) >= 30:
        model_results = train_model(model_df, target="percent_resistant")
        if model_results:
            model_results["feature_importances"].to_csv(
                OUTPUT / "feature_importances.csv", index=False)
            with open(OUTPUT / "model_metrics.json", "w") as f:
                json.dump(model_results["metrics"], f, indent=2)
            pd.DataFrame({
                "y_actual": model_results["y_test"],
                "y_predicted": model_results["y_pred"],
            }).to_csv(OUTPUT / "predictions.csv", index=False)
            print(f"  Saved metrics → {OUTPUT}/model_metrics.json")

    # ── Step 4: Plots ────────────────────────────────────────────────────────
    print("\n── Step 4: Generating plots ─────────────────────────────────────────")
    try:
        make_plots(amr, feat_df, model_results, PLOTS)
    except Exception as e:
        print(f"  [WARN] Plotting error: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Pipeline complete.  Outputs saved to:", OUTPUT.resolve())
    print("  AMR table rows:   ", len(amr))
    print("  Countries:        ", amr["country"].nunique())
    print("  Organisms:        ", sorted(amr["organism"].unique()))
    print("  Antibiotics:      ", sorted(amr["antibiotic"].unique()))
    print("  Matrix groups:    ", sorted(amr["matrix"].unique()))
    print("=" * 70)


if __name__ == "__main__":
    main()
