"""
fetch_abstracts_ncbi.py
-----------------------
Fetches PubMed abstracts relevant to AMR surveillance using NCBI E-utilities
and saves them to data/abstracts.jsonl for ingestion into ChromaDB.

Usage:
    python3 fetch_abstracts_ncbi.py
    python3 fetch_abstracts_ncbi.py --query "Salmonella resistance Europe" --max 500
"""

import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from config import NCBI_API_KEY, NCBI_EMAIL

from Bio import Entrez

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fetch_abstracts")

# ── Output path ───────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
OUTPUT_PATH = _ROOT / "data" / "abstracts.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Default search queries ────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "antimicrobial resistance Europe surveillance",
    "EARS-Net resistance trends Europe",
    "Salmonella antimicrobial resistance Europe",
    "Campylobacter resistance fluoroquinolone Europe",
    "E. coli ESBL resistance Europe",
    "MRSA Staphylococcus aureus Europe surveillance",
    "Enterococcus vancomycin resistance Europe",
    "One Health antimicrobial resistance zoonotic",
    "carbapenem resistant Klebsiella Europe",
    "AMR food chain animal resistance Europe",
]


def fetch_pmids(query: str, max_results: int = 200) -> list[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    result = Entrez.read(handle)
    handle.close()
    return result.get("IdList", [])


def fetch_records(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    handle  = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    parsed  = Entrez.read(handle)
    handle.close()
    records = []
    for article in parsed.get("PubmedArticle", []):
        try:
            med      = article["MedlineCitation"]
            pmid     = str(med["PMID"])
            title    = str(med["Article"]["ArticleTitle"])
            abstract_obj = med["Article"].get("Abstract", {})
            abstract_parts = abstract_obj.get("AbstractText", [""])
            abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""
            year     = str(
                med["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", "")
            )
            authors_list = med["Article"].get("AuthorList", [])
            authors = ", ".join(
                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                for a in authors_list[:5]
                if "LastName" in a
            )
            journal = str(med["Article"]["Journal"].get("Title", ""))
            if not abstract:
                continue
            records.append({
                "pmid":     pmid,
                "title":    title,
                "abstract": abstract,
                "year":     year,
                "authors":  authors,
                "journal":  journal,
            })
        except Exception as e:
            log.warning(f"Skipping article: {e}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for AMR Sentinel")
    parser.add_argument("--query",  type=str, help="Custom search query (overrides defaults)")
    parser.add_argument("--max",    type=int, default=200, help="Max results per query")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSONL path")
    args = parser.parse_args()

    # Configure Entrez
    if not NCBI_EMAIL:
        log.warning("NCBI_EMAIL not set in .env — NCBI may rate-limit requests")
    Entrez.email   = NCBI_EMAIL or "anonymous@example.com"
    Entrez.api_key = NCBI_API_KEY or None

    queries = [args.query] if args.query else DEFAULT_QUERIES
    output  = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_pmids = set()
    all_records = []

    for query in queries:
        log.info(f"Searching: {query!r}")
        pmids = fetch_pmids(query, max_results=args.max)
        new   = [p for p in pmids if p not in all_pmids]
        log.info(f"  Found {len(pmids)} PMIDs, {len(new)} new")
        all_pmids.update(new)

        if new:
            records = fetch_records(new)
            log.info(f"  Fetched {len(records)} records with abstracts")
            all_records.extend(records)

    # Deduplicate by PMID
    seen     = set()
    unique   = []
    for r in all_records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            unique.append(r)

    log.info(f"Writing {len(unique)} unique records to {output}")
    with open(output, "w", encoding="utf-8") as f:
        for rec in unique:
            f.write(json.dumps(rec) + "\n")

    log.info("Done.")


if __name__ == "__main__":
    main()
