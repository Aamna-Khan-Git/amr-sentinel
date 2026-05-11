"""
agents/data_agent.py
---------------------
Routes structured AMR questions to the appropriate SQLite query helper
and returns a JSON-serialisable summary dict.
"""

import logging
from typing import Optional

from db_setup import (
    query_trend,
    query_country_comparison,
    query_top_resistant_pairs,
    list_distinct_values,
    DB_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_data_query(
    intent: str,
    organism: Optional[str] = None,
    antibiotic: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
    db_path: str = DB_PATH,
) -> dict:
    """
    Dispatch a structured data query based on *intent* and return a summary dict.

    Parameters
    ----------
    intent : str
        One of ``"trend"``, ``"compare"``, ``"top_resistant"``, ``"list_values"``.
    organism : str, optional
        Pathogen name (e.g. ``"Escherichia coli"``).
    antibiotic : str, optional
        Antibiotic name (e.g. ``"ciprofloxacin"``).
    country : str, optional
        European country name.
    year : int, optional
        Calendar year (used as a filter or anchor for comparisons).
    limit : int
        Maximum rows for ``"top_resistant"`` queries.
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    dict
        Keys: ``intent``, ``query_params``, ``rows``, ``row_count``, ``error``.
    """
    query_params = {
        "intent":    intent,
        "organism":  organism,
        "antibiotic": antibiotic,
        "country":   country,
        "year":      year,
        "limit":     limit,
    }

    try:
        rows = _dispatch(intent, organism, antibiotic, country, year, limit, db_path)
        return {
            "intent":      intent,
            "query_params": query_params,
            "rows":        rows,
            "row_count":   len(rows),
            "error":       None,
        }
    except Exception as exc:
        logger.error("data_agent error [intent=%s]: %s", intent, exc, exc_info=True)
        return {
            "intent":      intent,
            "query_params": query_params,
            "rows":        [],
            "row_count":   0,
            "error":       str(exc),
        }


def get_available_values(field: str, db_path: str = DB_PATH) -> list[str]:
    """
    Convenience wrapper: return distinct values for a schema field.

    Useful for intent-parsing (autocomplete / fuzzy matching).
    """
    return list_distinct_values(field, db_path)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _dispatch(
    intent: str,
    organism: Optional[str],
    antibiotic: Optional[str],
    country: Optional[str],
    year: Optional[int],
    limit: int,
    db_path: str,
) -> list[dict]:
    """Route to the correct query helper."""
    if intent == "trend":
        _require(organism, "organism", intent)
        _require(antibiotic, "antibiotic", intent)
        return query_trend(organism, antibiotic, country=country, db_path=db_path)  # type: ignore[arg-type]

    elif intent == "compare":
        _require(organism, "organism", intent)
        _require(antibiotic, "antibiotic", intent)
        return query_country_comparison(organism, antibiotic, year=year, db_path=db_path)  # type: ignore[arg-type]

    elif intent == "top_resistant":
        return query_top_resistant_pairs(country=country, year=year, limit=limit, db_path=db_path)

    elif intent == "list_values":
        # organism is re-used as the field name here for simplicity
        field = organism or "country"
        values = list_distinct_values(field, db_path)
        return [{"value": v} for v in values]

    else:
        raise ValueError(
            f"Unknown intent '{intent}'. "
            "Supported: 'trend', 'compare', 'top_resistant', 'list_values'."
        )


def _require(value: Optional[str], name: str, intent: str) -> None:
    """Raise a descriptive error if a required parameter is missing."""
    if not value:
        raise ValueError(f"Parameter '{name}' is required for intent='{intent}'.")
