import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5-20251101"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are AMR Sentinel, an expert assistant in antimicrobial resistance (AMR)
surveillance for European public health and veterinary contexts.

Your task is to answer the user's question by synthesising:
1. Structured surveillance data from the European AMR monitoring network (EARS-Net / EFSA).
2. Relevant PubMed abstracts retrieved by semantic search.

Structure EVERY response in these sections:

### Summary
Brief 2-3 sentence overview of the resistance situation.

### Data Findings
Quantitative findings from surveillance data [DATA]. Include specific percentages,
country comparisons, and trends. Note sample sizes and limitations.

### Literature Insights
Key findings from retrieved abstracts [Rank N, PMID XXXXX]. Highlight mechanisms,
risk factors, and recent developments.

### Recommended Actions
This is the most important section. Provide specific, actionable recommendations
tailored to the resistance pattern observed. Include:
- **Clinical actions**: empirical therapy adjustments, stewardship interventions,
  diagnostic recommendations (e.g. culture before treatment, avoid X antibiotic
  empirically in Y country due to Z% resistance)
- **Public health actions**: surveillance priorities, outbreak response triggers,
  cross-border reporting needs
- **Veterinary/food chain actions**: if zoonotic pathogens are involved (Salmonella,
  Campylobacter, E. coli), recommend farm-level, slaughter hygiene, or food safety measures
- **Policy actions**: antibiotic formulary changes, prescribing restrictions,
  national action plan priorities

### Limitations & Uncertainties
Data gaps, sample size issues, surveillance methodology differences.

Guidelines:
- Be factual and cite sources: [DATA] for surveillance data, [Rank N, PMID XXXXX] for literature.
- Recommended Actions must be specific to the resistance levels found — e.g. if resistance
  exceeds 50%, flag as critical and recommend avoiding that antibiotic empirically.
- Use plain language suitable for public health professionals and veterinarians.
- If resistance is >50% flag as CRITICAL, 25-50% as HIGH, 10-25% as MODERATE, <10% as LOW.
"""


def generate_narrative(
    user_question: str,
    data_summary: dict,
    literature_context: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    api_key: Optional[str] = None,
) -> dict:
    try:
        from config import ANTHROPIC_API_KEY as _config_key
    except ImportError:
        _config_key = ""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "") or _config_key
    if not key:
        return _error_result("ANTHROPIC_API_KEY not set.")

    user_message = _build_user_message(user_question, data_summary, literature_context)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return {
            "answer":        answer_text,
            "model":         response.model,
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "error":         None,
        }
    except anthropic.APIStatusError as exc:
        logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return _error_result(f"API error {exc.status_code}: {exc.message}")
    except Exception as exc:
        logger.error("Unexpected error in narrative_agent: %s", exc, exc_info=True)
        return _error_result(str(exc))


def _build_user_message(question: str, data_summary: dict, literature_context: str) -> str:
    data_block = _format_data_summary(data_summary)
    return f"""## User Question
{question}

---

## Surveillance Data [DATA]

{data_block}

---

## Relevant PubMed Abstracts

{literature_context}

---

Please provide a careful, evidence-grounded answer to the question above.
Cite [DATA] for surveillance figures and [Rank N, PMID XXXXX] for literature.
Note any limitations or gaps in the evidence.
"""


def _format_data_summary(data_summary: dict) -> str:
    if data_summary.get("error"):
        return f"Data query error: {data_summary['error']}"

    params    = data_summary.get("query_params", {})
    rows      = data_summary.get("rows", [])
    row_count = data_summary.get("row_count", 0)

    header_parts = []
    for field in ["organism", "antibiotic", "country", "year"]:
        if params.get(field):
            header_parts.append(f"{field.capitalize()}: {params[field]}")

    header           = " | ".join(header_parts) if header_parts else "All records"
    display_rows     = rows[:20]
    truncation_note  = f"\n[Showing 20 of {row_count} rows]" if row_count > 20 else ""
    rows_json        = json.dumps(display_rows, indent=2)

    return (
        f"Query intent: {params.get('intent', 'unknown')}\n"
        f"Parameters: {header}\n"
        f"Total rows returned: {row_count}{truncation_note}\n\n"
        f"```json\n{rows_json}\n```"
    )


def _error_result(msg: str) -> dict:
    return {
        "answer":        f"[Narrative generation failed: {msg}]",
        "model":         None,
        "input_tokens":  0,
        "output_tokens": 0,
        "error":         msg,
    }
