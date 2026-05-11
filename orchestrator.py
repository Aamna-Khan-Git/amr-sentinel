import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import anthropic

from agents.data_agent import run_data_query, get_available_values
from agents.literature_agent import run_literature_search, build_literature_context
from agents.narrative_agent import generate_narrative
from db_setup import DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

INTENT_SYSTEM = """You are a parameter extractor for an AMR surveillance system.
Given a natural-language question, return ONLY a valid JSON object with these keys:

{
  "intent":     "<trend|compare|top_resistant|list_values>",
  "organism":   "<must be exact value from list below or null>",
  "antibiotic": "<must be exact value from list below or null>",
  "country":    "<country name or null>",
  "year":       <integer year or null>,
  "lit_query":  "<3-8 word semantic search phrase for PubMed>"
}

Intent definitions:
- trend         -> resistance percentage over time (requires organism + antibiotic)
- compare       -> resistance across countries (requires organism + antibiotic)
- top_resistant -> which organism/antibiotic pairs have highest resistance
- list_values   -> enumerate available countries/organisms/antibiotics

EXACT organism values (use verbatim):
C. coli, C. jejuni, E. coli, E. coli (ESBL/AmpC), E. faecalis, E. faecium,
S. Derby, S. Enteritidis, S. Infantis, S. Kentucky, S. Typhimurium,
S. Typhimurium (mono), Salmonella spp.

EXACT antibiotic codes (use verbatim):
AMC, AMK, AMP, AZM, AmpC_PHENO, CAZ, CHL, CIP, COL, CTX,
DPT, ERY, ESBL_AmpC_PHENO, ESBL_PHENO, ETP, GEN, LZD, MEM,
NAL, QDA, SMX, SXT, TEC, TET, TGC, TMP, VAN

Mapping hints:
- ciprofloxacin / fluoroquinolone  -> CIP
- ampicillin                       -> AMP
- tetracycline                     -> TET
- gentamicin                       -> GEN
- colistin / polymyxin             -> COL
- ceftazidime / cephalosporin      -> CAZ
- meropenem                        -> MEM
- ertapenem                        -> ETP
- vancomycin                       -> VAN
- azithromycin / macrolide         -> AZM
- chloramphenicol                  -> CHL
- trimethoprim                     -> TMP
- co-trimoxazole / sulfa           -> SXT
- sulfamethoxazole                 -> SMX
- linezolid                        -> LZD
- tigecycline                      -> TGC
- nalidixic acid                   -> NAL
- amoxicillin-clavulanate          -> AMC
- amikacin                         -> AMK
- ESBL / extended spectrum         -> ESBL_PHENO
- Campylobacter                    -> C. jejuni
- Salmonella (no serovar)          -> Salmonella spp.
- Salmonella Typhimurium           -> S. Typhimurium
- Salmonella Enteritidis           -> S. Enteritidis
- Enterococcus faecalis            -> E. faecalis
- Enterococcus faecium             -> E. faecium

Return ONLY JSON. No explanations."""


def ask(
    question: str,
    k_literature: int = 5,
    db_path: str = DB_PATH,
    api_key: Optional[str] = None,
) -> dict:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _error_payload(question, "ANTHROPIC_API_KEY not set.")

    logger.info("Parsing intent for: %r", question)
    parsed = _parse_intent(question, api_key=key)
    if parsed.get("_error"):
        return _error_payload(question, f"Intent parsing failed: {parsed['_error']}")

    intent     = parsed.get("intent", "top_resistant")
    organism   = parsed.get("organism")
    antibiotic = parsed.get("antibiotic")
    country    = parsed.get("country")
    year       = parsed.get("year")
    lit_query  = parsed.get("lit_query", question)

    logger.info("Intent=%s  organism=%s  antibiotic=%s  country=%s  year=%s",
                intent, organism, antibiotic, country, year)

    logger.info("Running data query ...")
    data_result = run_data_query(
        intent=intent,
        organism=organism,
        antibiotic=antibiotic,
        country=country,
        year=year,
        db_path=db_path,
    )

    logger.info("Searching literature for: %r", lit_query)
    lit_result = run_literature_search(query=lit_query, k=k_literature)
    lit_context = build_literature_context(lit_result["hits"])

    logger.info("Generating narrative ...")
    narrative = generate_narrative(
        user_question=question,
        data_summary=data_result,
        literature_context=lit_context,
        api_key=key,
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


def _parse_intent(question: str, api_key: str) -> dict:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=INTENT_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        raw_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
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
