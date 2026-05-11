import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from literature_store import search_literature, collection_count, CHROMA_DIR

logger = logging.getLogger(__name__)


def run_literature_search(
    query: str,
    k: int = 5,
    year_filter: Optional[int] = None,
    chroma_dir: str = CHROMA_DIR,
) -> dict:
    try:
        store_size = collection_count(chroma_dir)
        if store_size == 0:
            logger.warning("ChromaDB collection is empty.")
            return _empty_result(query, k, store_size, warning="Collection is empty.")
        hits = search_literature(query, k=k, chroma_dir=chroma_dir, year_filter=year_filter)
        formatted = [_format_hit(h, rank=i + 1) for i, h in enumerate(hits)]
        return {"query": query, "k": k, "hits": formatted,
                "hit_count": len(formatted), "store_size": store_size, "error": None}
    except Exception as exc:
        logger.error("literature_agent error: %s", exc, exc_info=True)
        return {"query": query, "k": k, "hits": [], "hit_count": 0,
                "store_size": 0, "error": str(exc)}


def build_literature_context(hits: list[dict], max_chars: int = 3000) -> str:
    if not hits:
        return "No relevant literature found in the store."
    parts, total_chars = [], 0
    for hit in hits:
        entry = (f"[{hit['rank']}] PMID {hit['pmid']} ({hit['year']}) - {hit['title']}\n"
                 f"    {hit['authors']}\n    {hit['journal']}\n    Snippet: {hit['snippet']}\n")
        if total_chars + len(entry) > max_chars:
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n".join(parts)


def _format_hit(raw: dict, rank: int) -> dict:
    return {"rank": rank, "pmid": raw.get("pmid", ""), "title": raw.get("title", ""),
            "authors": raw.get("authors", ""), "journal": raw.get("journal", ""),
            "year": raw.get("year", ""), "snippet": raw.get("snippet", ""),
            "distance": raw.get("distance", 0.0), "citation": _make_citation(raw)}


def _make_citation(hit: dict) -> str:
    parts = [f"{hit.get('authors', 'Unknown')} ({hit.get('year', 'n.d.')}). {hit.get('title', 'Untitled')}."]
    if hit.get("journal"):
        parts.append(hit["journal"] + ".")
    if hit.get("pmid"):
        parts.append(f"PMID: {hit['pmid']}")
    return " ".join(parts)


def _empty_result(query: str, k: int, store_size: int, warning: str = "") -> dict:
    return {"query": query, "k": k, "hits": [], "hit_count": 0,
            "store_size": store_size, "error": warning or None}
