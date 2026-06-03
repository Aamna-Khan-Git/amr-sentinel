"""
literature_store.py
-------------------
Ingests PubMed abstracts into ChromaDB and provides semantic search.
ChromaDB path is read from config (CHROMA_PATH env var).
"""

import json
import csv
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

CHROMA_DIR      = CHROMA_PATH
COLLECTION_NAME = "pubmed_amr"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# Add at the top of literature_store.py after imports
_COLLECTION_CACHE = {}

def _get_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    if chroma_dir in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[chroma_dir]
    client = chromadb.PersistentClient(path=chroma_dir)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col    = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    _COLLECTION_CACHE[chroma_dir] = col
    return col

def _get_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _load_records_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)
    return records


def _load_records_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _record_to_document(rec: dict) -> tuple[str, str, dict]:
    pmid     = str(rec.get("pmid", "")).strip()
    title    = str(rec.get("title", "")).strip()
    abstract = str(rec.get("abstract", "")).strip()
    if not pmid:
        raise ValueError("Record missing 'pmid' field.")
    if not abstract:
        raise ValueError(f"Record pmid={pmid} has no abstract.")
    document_text = f"{title}\n\n{abstract}" if title else abstract
    metadata = {
        "pmid":    pmid,
        "title":   title,
        "authors": str(rec.get("authors", "")),
        "journal": str(rec.get("journal", "")),
        "year":    str(rec.get("year", "")),
    }
    return pmid, document_text, metadata


def ingest_literature(
    path: str,
    chroma_dir: str = CHROMA_DIR,
    batch_size: int = 64,
) -> int:
    if not Path(path).exists():
        raise FileNotFoundError(f"Literature file not found: {path}")
    ext = Path(path).suffix.lower()
    if ext in {".jsonl", ".json"}:
        records = _load_records_jsonl(path)
    elif ext == ".csv":
        records = _load_records_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .jsonl or .csv")

    logger.info("Loaded %d raw records from %s", len(records), path)
    collection = _get_collection(chroma_dir)
    ids, documents, metadatas = [], [], []
    skipped = indexed = 0

    for rec in records:
        try:
            doc_id, doc_text, meta = _record_to_document(rec)
        except ValueError as exc:
            logger.warning("Skipping record: %s", exc)
            skipped += 1
            continue
        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)
        if len(ids) >= batch_size:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            indexed += len(ids)
            logger.info("Indexed batch: %d docs (total: %d)", len(ids), indexed)
            ids, documents, metadatas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        indexed += len(ids)

    logger.info("Done. Indexed: %d | Skipped: %d | Store total: %d",
                indexed, skipped, collection.count())
    return indexed


def ingest_records(records: list[dict], chroma_dir: str = CHROMA_DIR, batch_size: int = 64) -> int:
    """
    Ingest a list of record dicts directly (used by pdf_ingest.py).
    Each record must have: pmid, title, abstract, year.
    """
    collection = _get_collection(chroma_dir)
    ids, documents, metadatas = [], [], []
    skipped = indexed = 0

    for rec in records:
        try:
            doc_id, doc_text, meta = _record_to_document(rec)
        except ValueError as exc:
            logger.warning("Skipping record: %s", exc)
            skipped += 1
            continue
        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)
        if len(ids) >= batch_size:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            indexed += len(ids)
            ids, documents, metadatas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        indexed += len(ids)

    logger.info("ingest_records: indexed=%d skipped=%d total=%d",
                indexed, skipped, collection.count())
    return indexed


def search_literature(
    query: str,
    k: int = 5,
    chroma_dir: str = CHROMA_DIR,
    year_filter: Optional[int] = None,
) -> list[dict]:
    collection = _get_collection(chroma_dir)
    where      = {"year": {"$gte": str(year_filter)}} if year_filter is not None else None
    results    = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        snippet = doc.split("\n\n", 1)[-1][:400]
        hits.append({
            "pmid":     meta.get("pmid", ""),
            "title":    meta.get("title", ""),
            "authors":  meta.get("authors", ""),
            "journal":  meta.get("journal", ""),
            "year":     meta.get("year", ""),
            "snippet":  snippet,
            "distance": round(float(dist), 4),
        })
    return hits


def collection_count(chroma_dir: str = CHROMA_DIR) -> int:
    return _get_collection(chroma_dir).count()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python literature_store.py <path_to_abstracts.jsonl>")
        sys.exit(1)
    ingest_literature(sys.argv[1])
    print(f"Store now contains {collection_count()} documents.")
