"""
agents/narrative_agent.py
--------------------------
Generates clinical AMR narratives using the configured LLM provider.
Severity labels (CRITICAL/HIGH/MODERATE/LOW) are applied as post-processing
in Python rather than relying on the LLM, ensuring 100% accuracy.
"""

import re
import json
import logging
import sys
import requests
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import API_KEY, API_BASE_URL, NARRATIVE_MODEL

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096

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
  diagnostic recommendations
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
- Recommended Actions must be specific to the resistance levels found.
- Use plain language suitable for public health professionals and veterinarians.

STRICT DATA RULES:
- ONLY report countries, values, and organisms present in the data rows provided.
- Do NOT invent averages, geographic trends, or countries not in the data.
- If the dataset has fewer than 10 rows, state that explicitly rather than padding.
- Every specific number must come directly from the provided rows.
- source_type values: 'human'=clinical cases, 'animal_pig'=pig farms,
  'meat_pork'=pork meat, 'meat_broiler'=broiler meat."""


# ── Severity post-processor ───────────────────────────────────────────────────

def _get_severity(pct: float) -> str:
    if pct >= 50:   return "CRITICAL"
    if pct >= 25:   return "HIGH"
    if pct >= 10:   return "MODERATE"
    return "LOW"


def apply_severity_labels(text: str) -> str:
    """
    Post-process the narrative text to ensure every resistance percentage
    has the correct severity label. This guarantees accuracy regardless of
    what the LLM outputs.

    Replaces existing labels if wrong, adds label if missing.
    Pattern matches: 52.1%, 52.1% (HIGH), 52.1% (CRITICAL) etc.
    """
    # Match percentage followed by optional existing label
    pattern = re.compile(
        r'(\d+\.?\d*)\s*%\s*(?:\((CRITICAL|HIGH|MODERATE|LOW)\))?',
        re.IGNORECASE
    )

    def replace_label(m):
        pct       = float(m.group(1))
        correct   = _get_severity(pct)
        return f"{m.group(1)}% ({correct})"

    return pattern.sub(replace_label, text)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(model: str, system: str, user: str, max_tokens: int = 2048) -> str:
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
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ── Main function ─────────────────────────────────────────────────────────────

def generate_narrative(
    user_question: str,
    data_summary: dict,
    literature_context: str,
    model: str = NARRATIVE_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    if not API_KEY:
        return _error_result("API_KEY not set in .env")

    user_message = _build_user_message(user_question, data_summary, literature_context)

    try:
        raw_answer  = _call_llm(model, SYSTEM_PROMPT, user_message, max_tokens=max_tokens)
        answer_text = apply_severity_labels(raw_answer)
        answer_text = apply_severity_labels(raw_answer)
        answer_text = re.sub(r'PMID[:\s]+ECDC_\w+', '[ECDC surveillance report]', answer_text)
        answer_text = re.sub(r'PMID[:\s]+[A-Z]{2,}_\w+', '[surveillance report]', answer_text)
        # Apply correct severity labels as post-processing
        return {
            "answer":        answer_text,
            "model":         model,
            "input_tokens":  0,
            "output_tokens": 0,
            "error":         None,
        }
    except Exception as exc:
        logger.error("Narrative generation error: %s", exc, exc_info=True)
        return _error_result(str(exc))


# ── Message builders ──────────────────────────────────────────────────────────

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
Note any limitations or gaps in the evidence."""


def _format_data_summary(data_summary: dict) -> str:
    if data_summary.get("error"):
        return f"Data query error: {data_summary['error']}"

    params    = data_summary.get("query_params", {})
    rows      = data_summary.get("rows", [])
    row_count = data_summary.get("row_count", 0)

    header_parts = [
        f"{f.capitalize()}: {params[f]}"
        for f in ["organism", "antibiotic", "country", "year"]
        if params.get(f)
    ]
    header          = " | ".join(header_parts) if header_parts else "All records"
    display_rows    = rows[:20]
    truncation_note = f"\n[Showing 20 of {row_count} rows]" if row_count > 20 else ""

    # Plain text format — more reliable than JSON for smaller models
    plain_summary = ""
    for row in display_rows:
        parts = []
        if "country"    in row: parts.append(f"Country={row['country']}")
        if "organism"   in row: parts.append(f"Organism={row['organism']}")
        if "antibiotic" in row: parts.append(f"Antibiotic={row['antibiotic']}")
        if "avg_pct_resistant" in row:
            pct = row["avg_pct_resistant"]
            parts.append(f"Resistance={pct}%")
        if "source_type" in row:
            src = {
                "human":          "human clinical",
                "animal_pig":     "pig farms",
                "meat_pork":      "pork meat",
                "meat_broiler":   "broiler meat",
                "animal_broiler": "broiler farms",
                "animal_cattle":  "cattle farms",
                "meat_cattle":    "beef meat",
            }.get(row["source_type"], row["source_type"])
            if row.get("country") == "EU/EEA":
                src += " (EU/EEA aggregate, not country-specific)"
            parts.append(f"Source={src}")
            parts.append(f"Source={src}")
        if "total_isolates" in row and row["total_isolates"] and int(row["total_isolates"]) > 0:
            parts.append(f"Isolates={row['total_isolates']}")
        if "year" in row and row.get("year"):
            parts.append(f"Year={row['year']}")
        plain_summary += "  - " + ", ".join(parts) + "\n"

    return (
        f"Query intent: {params.get('intent', 'unknown')} "
        f"(showing top resistant combinations for this query)\n"
        f"Parameters: {header}\n"
        f"Total rows returned: {row_count}{truncation_note}\n"
        f"Source types: 'human'=clinical cases, 'animal_pig'=pig farms, "
        f"'meat_pork'=pork meat, 'meat_broiler'=broiler meat\n\n"
        f"RESISTANCE DATA (use these exact numbers in your answer):\n"
        f"{plain_summary}"
    )


def _error_result(msg: str) -> dict:
    return {
        "answer":        f"[Narrative generation failed: {msg}]",
        "model":         None,
        "input_tokens":  0,
        "output_tokens": 0,
        "error":         msg,
    }
