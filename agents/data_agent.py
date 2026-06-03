"""
agents/data_agent.py
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


def run_data_query(
    intent: str,
    organism: Optional[str] = None,
    antibiotic: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
    db_path: str = DB_PATH,
) -> dict:
    query_params = {
        "intent":     intent,
        "organism":   organism,
        "antibiotic": antibiotic,
        "country":    country,
        "year":       year,
        "limit":      limit,
    }
    try:
        rows = _dispatch(intent, organism, antibiotic, country, year, limit, db_path)
        return {
            "intent":       intent,
            "query_params": query_params,
            "rows":         rows,
            "row_count":    len(rows),
            "error":        None,
        }
    except Exception as exc:
        logger.error("data_agent error [intent=%s]: %s", intent, exc, exc_info=True)
        return {
            "intent":       intent,
            "query_params": query_params,
            "rows":         [],
            "row_count":    0,
            "error":        str(exc),
        }


def get_available_values(field: str, db_path: str = DB_PATH) -> list[str]:
    return list_distinct_values(field, db_path)


def _dispatch(
    intent: str,
    organism: Optional[str],
    antibiotic: Optional[str],
    country: Optional[str],
    year: Optional[int],
    limit: int,
    db_path: str,
) -> list[dict]:
    if intent == "trend":
        if not organism or not antibiotic:
            return query_top_resistant_pairs(country=country, year=year, limit=limit, db_path=db_path)
        rows = query_trend(organism, antibiotic, country=country, db_path=db_path)
        if len(rows) <= 1:
            # Only one year of data — fall back to country comparison for richer answer
            compare_rows = query_country_comparison(organism, antibiotic, year=year, db_path=db_path)
            if compare_rows:
                return compare_rows
        return rows

    elif intent == "compare":
        if not organism or not antibiotic:
            # Broad question — fall back to top resistant pairs
            return query_top_resistant_pairs(country=country, year=year, limit=limit, db_path=db_path)
        return query_country_comparison(organism, antibiotic, year=year, db_path=db_path)

    elif intent == "top_resistant":
        return query_top_resistant_pairs(country=country, year=year, limit=limit, db_path=db_path)

    elif intent == "list_values":
        allowed = {"country", "organism", "antibiotic", "year", "source_type"}
        field = organism or "country"
        if field not in allowed:
            field = "country"
        values = list_distinct_values(field, db_path)
        return [{"value": v} for v in values]

    else:
        raise ValueError(
            f"Unknown intent '{intent}'. "
            "Supported: 'trend', 'compare', 'top_resistant', 'list_values'."
        )


def _require(value: Optional[str], name: str, intent: str) -> None:
    if not value:
        raise ValueError(f"Parameter '{name}' is required for intent='{intent}'.")
