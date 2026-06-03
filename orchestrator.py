import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import API_KEY, API_BASE_URL, INTENT_MODEL, DATABASE_PATH
from agents.data_agent import run_data_query
from agents.literature_agent import run_literature_search, build_literature_context
from agents.narrative_agent import generate_narrative

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

NON_COUNTRIES = {
    "europe", "european", "eu", "eu/eea", "european union",
    "eastern europe", "western europe", "northern europe", "southern europe",
    "scandinavia", "balkans", "mediterranean", "globally", "worldwide", "world",
}

INTENT_SYSTEM = """You are an intelligent parameter extractor for an AMR (antimicrobial resistance) surveillance system.

Your job is to READ the user's question carefully, UNDERSTAND what they are actually asking for, and INFER the correct JSON parameters using logic — not keyword matching.

Return ONLY a valid JSON object:
{
  "intent":     "<trend|compare|top_resistant|list_values>",
  "organism":   "<exact value from list or null>",
  "antibiotic": "<exact code from list or null>",
  "country":    "<country name or null>",
  "year":       <integer or null>,
  "lit_query":  "<3-8 word semantic search phrase>"
}

INTENT — infer from what the user WANTS to know:

trend = user wants to see change over time
  Logic: question contains time words (over time, increasing, decreasing, trend, years, changing, grown, risen)
  Examples:
    "How has CIP resistance changed?" -> trend
    "Is resistance increasing?" -> trend
    "Trend of ESBL over the years" -> trend

compare = user wants to compare resistance levels across countries, organisms, or sources
  Logic: user wants to know WHICH is highest/worst/safest, or wants country/source breakdown
  Examples:
    "Which countries have highest resistance?" -> compare
    "Which antibiotics should I avoid in Germany?" -> compare, country=Germany
    "Which meat is safest in Bulgaria?" -> compare, country=Bulgaria
    "I am from France, what resistance should I worry about?" -> compare, country=France
    "Compare resistance across Europe" -> compare
    "What is the AMR situation in Italy?" -> compare, country=Italy

top_resistant = user wants an overall ranking without specifying organism or antibiotic
  Logic: broad question about worst/highest resistance combinations, no specific pathogen mentioned
  Examples:
    "What are the most resistant combinations?" -> top_resistant
    "Show me top 10 resistance" -> top_resistant
    "Which bugs and drugs are worst?" -> top_resistant

list_values = ONLY when user explicitly asks what data is available in the system
  Logic: user is asking about the database itself, not about resistance
  Examples:
    "What organisms are in the database?" -> list_values
    "List available countries" -> list_values
  NEVER use list_values for resistance questions.

ORGANISM — extract only if explicitly mentioned, otherwise null:
  Do NOT guess or default. If not mentioned, return null.
  EXACT values: C. coli, C. jejuni, E. coli, E. coli (ESBL/AmpC), E. faecalis, E. faecium,
  S. Derby, S. Enteritidis, S. Infantis, S. Kentucky, S. Typhimurium,
  S. Typhimurium (mono), Salmonella spp.
  Aliases:
    Campylobacter (alone) -> C. jejuni
    Salmonella (no serovar) -> Salmonella spp.
    Salmonella Typhimurium -> S. Typhimurium
    Salmonella Enteritidis -> S. Enteritidis
    Enterococcus faecalis -> E. faecalis
    Enterococcus faecium -> E. faecium
    E. coli / Escherichia coli -> E. coli

ANTIBIOTIC — extract only if explicitly mentioned, otherwise null:
  Do NOT guess or default. If not mentioned, return null.
  EXACT codes: AMC, AMK, AMP, AZM, CAZ, CHL, CIP, COL, CTX, ERY,
  ESBL_PHENO, ETP, GEN, LZD, MEM, NAL, QDA, SMX, SXT, TET, TGC, TMP, VAN
  Aliases:
    ciprofloxacin / fluoroquinolone -> CIP
    ampicillin / amoxicillin -> AMP
    tetracycline -> TET
    vancomycin -> VAN
    gentamicin -> GEN
    colistin -> COL
    azithromycin -> AZM
    chloramphenicol -> CHL
    trimethoprim -> TMP
    meropenem -> MEM
    ESBL / extended spectrum -> ESBL_PHENO

COUNTRY — extract ONLY if a specific nation is mentioned:
  - "Europe", "European", "EU", "across Europe", "in Europe" -> null
  - "I am from Germany" -> Germany
  - "in Bulgaria" -> Bulgaria
  - "across European countries" -> null
  - "in Eastern Europe" -> null
  - Only extract specific nation states, never regions or continents

DATA AWARENESS:
  - "increasing", "decreasing", "over time", "over the years" -> trend intent
  - The system has 2015 animal/food data and 2020-2024 human data
  - When country=null the system queries all countries automatically

lit_query — 3-8 word PubMed search phrase:
  "I am from Germany, which antibiotics to avoid" -> "antimicrobial resistance Germany food chain"
  "Which meat is safest in Bulgaria" -> "Salmonella AMR Bulgaria meat"
"Which country has the best/worst AMR" -> compare, all organisms, no country filter
"Which country is safest" -> compare
Return ONLY the JSON object. No explanation. No markdown."""


def _call_llm(model: str, system: str, user: str, max_tokens: int = 256) -> str:
    response = requests.post(
        f"{API_BASE_URL}/api/chat",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model":   model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {"num_predict": max_tokens},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def ask(question: str, k_literature: int = 5, db_path: str = DATABASE_PATH) -> dict:
    if not API_KEY:
        return _error_payload(question, "API_KEY not set in .env")

    logger.info("Parsing intent for: %r", question)
    parsed = _parse_intent(question)

    if parsed.get("_error"):
        return _error_payload(question, f"Intent parsing failed: {parsed['_error']}")

    # Normalise non-country geographic terms to null
    if (parsed.get("country") or "").lower() in NON_COUNTRIES:
        parsed["country"] = None

    intent     = parsed.get("intent", "top_resistant")
    organism   = parsed.get("organism")
    antibiotic = parsed.get("antibiotic")
    country    = parsed.get("country")
    year       = parsed.get("year")
    lit_query  = parsed.get("lit_query", question)

    logger.info("Intent=%s  organism=%s  antibiotic=%s  country=%s  year=%s",
                intent, organism, antibiotic, country, year)

    data_result = run_data_query(
        intent=intent, organism=organism, antibiotic=antibiotic,
        country=country, year=year, db_path=db_path,
    )

    lit_result  = run_literature_search(query=lit_query, k=k_literature)
    lit_context = build_literature_context(lit_result["hits"])

    narrative = generate_narrative(
        user_question=question,
        data_summary=data_result,
        literature_context=lit_context,
    )

    return {
        "question":           question,
        "intent":             intent,
        "parsed_params":      parsed,
        "data":               data_result,
        "literature":         lit_result,
        "answer":             narrative["answer"],
        "narrative_metadata": {
            "model":         narrative.get("model"),
            "input_tokens":  narrative.get("input_tokens"),
            "output_tokens": narrative.get("output_tokens"),
        },
        "error": narrative.get("error"),
    }


def _parse_intent(question: str) -> dict:
    try:
        raw_text = _call_llm(INTENT_MODEL, INTENT_SYSTEM, question, max_tokens=256)
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse intent JSON: %s", exc)
        return {"_error": f"JSON decode: {exc}"}
    except Exception as exc:
        logger.error("Intent parsing error: %s", exc)
        return {"_error": str(exc)}


def _error_payload(question: str, msg: str) -> dict:
    return {
        "question":           question,
        "intent":             None,
        "parsed_params":      {},
        "data":               {"row_count": 0, "rows": [], "error": msg},
        "literature":         {"hits": [], "hit_count": 0, "store_size": 0},
        "answer":             f"[Error: {msg}]",
        "narrative_metadata": {},
        "error":              msg,
    }
