"""
Single-Agent RAG Baseline for AMR Sentinel

User Query
    -> Simple query understanding
    -> SQLite structured-data retrieval
    -> ChromaDB PubMed retrieval
    -> Combined evidence
    -> One LLM
    -> Final answer
"""

import logging
from typing import Optional

import requests

from config import API_KEY, API_BASE_URL, NARRATIVE_MODEL, DATABASE_PATH
from agents.data_agent import run_data_query
from literature_store import search_literature, collection_count, CHROMA_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# 1. LLM CALL
# =========================================================

def call_llm(prompt: str) -> str:

    if not API_KEY:
        raise RuntimeError("API_KEY is not configured.")

    if API_BASE_URL.rstrip("/").endswith("/v1"):

        url = f"{API_BASE_URL.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": NARRATIVE_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an antimicrobial resistance information assistant. "
                        "Answer only using the evidence provided. "
                        "Do not invent statistics, countries, organisms, antibiotics, "
                        "or study findings."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError(
        "This baseline expects an OpenAI-compatible /v1 API endpoint."
    )


# =========================================================
# 2. SIMPLE QUERY UNDERSTANDING
# =========================================================

def understand_query(question: str) -> dict:
    """
    Extract the basic parameters needed by the SQLite retrieval.

    This is NOT an agent.
    It is simply a structured query-understanding step.
    """

    prompt = f"""
Extract the following information from the AMR question.

Question:
{question}

Return ONLY valid JSON with these fields:

{{
    "intent": "trend | compare | top_resistant | list_values",
    "organism": "string or null",
    "antibiotic": "string or null",
    "country": "string or null",
    "year": "integer or null"
}}

Rules:

- Use "trend" when the question asks about change over time.
- Use "compare" when comparing countries.
- Use "top_resistant" when asking which organisms/antibiotics have high
or highest resistance.
- Use "list_values" when asking to list available countries, organisms,
antibiotics, etc.
- If a value is not explicitly present, use null.
"""

    raw = call_llm(prompt)

    import json

    try:
        result = json.loads(raw)

        return {
            "intent": result.get("intent", "top_resistant"),
            "organism": result.get("organism"),
            "antibiotic": result.get("antibiotic"),
            "country": result.get("country"),
            "year": result.get("year"),
        }

    except Exception:
        logger.warning("Could not parse query parameters. Using defaults.")

        return {
            "intent": "top_resistant",
            "organism": None,
            "antibiotic": None,
            "country": None,
            "year": None,
        }


# =========================================================
# 3. SQLITE RETRIEVAL
# =========================================================

def retrieve_structured_data(params: dict):

    return run_data_query(
        intent=params["intent"],
        organism=params.get("organism"),
        antibiotic=params.get("antibiotic"),
        country=params.get("country"),
        year=params.get("year"),
        db_path=DATABASE_PATH,
    )


# =========================================================
# 4. PUBMED RETRIEVAL
# =========================================================

def retrieve_literature(question: str, k: int = 5):

    store_size = collection_count(CHROMA_DIR)

    if store_size == 0:
        return []

    return search_literature(
        query=question,
        k=k,
        chroma_dir=CHROMA_DIR,
    )


# =========================================================
# 5. FORMAT STRUCTURED DATA
# =========================================================

def format_structured_data(result: dict) -> str:

    rows = result.get("rows", [])

    if not rows:
        return "No structured surveillance data was retrieved."

    lines = []

    for i, row in enumerate(rows, start=1):
        lines.append(f"Data row {i}: {row}")

    return "\n".join(lines)


# =========================================================
# 6. FORMAT PUBMED DATA
# =========================================================

def format_literature(hits: list[dict]) -> str:

    if not hits:
        return "No relevant PubMed literature was retrieved."

    parts = []

    for i, hit in enumerate(hits, start=1):

        parts.append(
            f"""
Literature {i}:
PMID: {hit.get('pmid', '')}
Title: {hit.get('title', '')}
Authors: {hit.get('authors', '')}
Journal: {hit.get('journal', '')}
Year: {hit.get('year', '')}
Snippet: {hit.get('snippet', '')}
"""
        )

    return "\n".join(parts)


# =========================================================
# 7. SINGLE-AGENT RAG
# =========================================================

def single_agent_rag(
    question: str,
    k_literature: int = 5,
):

    logger.info("Running single-agent RAG baseline...")

    # -----------------------------------------
    # Step 1: Understand the question
    # -----------------------------------------

    params = understand_query(question)

    logger.info("Query parameters: %s", params)

    # -----------------------------------------
    # Step 2: Retrieve structured AMR data
    # -----------------------------------------

    data_result = retrieve_structured_data(params)

    # -----------------------------------------
    # Step 3: Retrieve PubMed literature
    # -----------------------------------------

    literature_hits = retrieve_literature(
        question,
        k=k_literature,
    )

    # -----------------------------------------
    # Step 4: Combine evidence
    # -----------------------------------------

    structured_context = format_structured_data(
        data_result
    )

    literature_context = format_literature(
        literature_hits
    )

    combined_context = f"""
================ STRUCTURED SURVEILLANCE DATA ================

{structured_context}


================ PUBMED LITERATURE ================

{literature_context}
"""

    # -----------------------------------------
    # Step 5: Generate answer
    # -----------------------------------------

    prompt = f"""
Answer the following AMR question using ONLY the retrieved evidence.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{combined_context}

Instructions:

1. Answer the question directly.
2. Use the structured surveillance data when relevant.
3. Use PubMed evidence when relevant.
4. Do not invent numerical values or factual claims.
5. Clearly distinguish surveillance data from literature evidence.
6. If the evidence does not contain enough information, say so.
"""

    answer = call_llm(prompt)

    return {
        "question": question,
        "query_parameters": params,
        "data": data_result,
        "literature": literature_hits,
        "answer": answer,
    }


# =========================================================
# 8. TEST
# =========================================================

if __name__ == "__main__":

    question = input("\nEnter your AMR question: ")

    result = single_agent_rag(question)

    print("\n" + "=" * 70)
    print("SINGLE-AGENT RAG BASELINE")
    print("=" * 70)

    print("\nQUESTION:")
    print(result["question"])

    print("\nQUERY PARAMETERS:")
    print(result["query_parameters"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSTRUCTURED DATA ROWS:")
    print(result["data"].get("row_count", 0))

    print("\nLITERATURE HITS:")
    print(len(result["literature"]))