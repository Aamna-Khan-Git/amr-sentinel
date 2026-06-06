"""
eval.py
-------
Factual evaluation of AMR Sentinel across four dimensions:
1. Intent + Entity Extraction Accuracy   (rule-based, free)
2. Citation Hallucination Rate           (rule-based, free)
3. Severity Label Accuracy               (rule-based, free)
4. Answer Quality                        (LLM judge, only fires on failures)

Run with:
    python3 eval.py
    python3 eval.py --verbose            # show answer excerpts
    python3 eval.py --output report.json # save full JSON report
"""

import json
import re
import sys
import argparse
import logging
import requests
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)

from orchestrator import ask

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# ============================================================
# EVALUATION DATASET  (10 questions — one per key scenario)
# Covers: 4 trend · 3 compare · 2 top_resistant · 1 hard/mixed
# ============================================================

EVAL_QUESTIONS = [
    # ── Trend ────────────────────────────────────────────────
    {
        "question":            "How has ciprofloxacin resistance in E. coli changed over time?",
        "expected_intent":     "trend",
        "expected_organism":   "E. coli",
        "expected_antibiotic": "CIP",
        "expected_country":    None,
    },
    {
        "question":            "Show me the trend of tetracycline resistance in Salmonella Typhimurium",
        "expected_intent":     "trend",
        "expected_organism":   "S. Typhimurium",
        "expected_antibiotic": "TET",
        "expected_country":    None,
    },
    {
        "question":            "How has ampicillin resistance in E. coli changed in Germany over the years?",
        "expected_intent":     "trend",
        "expected_organism":   "E. coli",
        "expected_antibiotic": "AMP",
        "expected_country":    "Germany",
    },
    {
        "question":            "Trend of ESBL resistance in E. coli over the years",
        "expected_intent":     "trend",
        "expected_organism":   "E. coli (ESBL/AmpC)",
        "expected_antibiotic": "ESBL_PHENO",
        "expected_country":    None,
    },
    # ── Compare ──────────────────────────────────────────────
    {
        "question":            "Compare ciprofloxacin resistance in E. coli across European countries",
        "expected_intent":     "compare",
        "expected_organism":   "E. coli",
        "expected_antibiotic": "CIP",
        "expected_country":    None,
    },
    {
        "question":            "Which countries have the highest tetracycline resistance in Salmonella?",
        "expected_intent":     "compare",
        "expected_organism":   "Salmonella spp.",
        "expected_antibiotic": "TET",
        "expected_country":    None,
    },
    {
        "question":            "Compare vancomycin resistance in Enterococcus faecium across countries",
        "expected_intent":     "compare",
        "expected_organism":   "E. faecium",
        "expected_antibiotic": "VAN",
        "expected_country":    None,
    },
    # ── Top resistant ─────────────────────────────────────────
    {
        "question":            "Which organism and antibiotic combinations have the highest resistance?",
        "expected_intent":     "top_resistant",
        "expected_organism":   None,
        "expected_antibiotic": None,
        "expected_country":    None,
    },
    {
        "question":            "Show me the top 10 highest resistance combinations",
        "expected_intent":     "top_resistant",
        "expected_organism":   None,
        "expected_antibiotic": None,
        "expected_country":    None,
    },
    # ── Hard / mixed ──────────────────────────────────────────
    {
        "question":            "Is fluoroquinolone resistance in Campylobacter increasing?",
        "expected_intent":     "trend",
        "expected_organism":   "C. jejuni",
        "expected_antibiotic": "CIP",
        "expected_country":    None,
    },
]

# ============================================================
# SEVERITY THRESHOLDS  (must match narrative_agent.py)
# ============================================================

SEVERITY_THRESHOLDS = [
    (50, "CRITICAL"),
    (25, "HIGH"),
    (10, "MODERATE"),
    (0,  "LOW"),
]

def expected_severity(pct: float) -> str:
    for threshold, label in SEVERITY_THRESHOLDS:
        if pct >= threshold:
            return label
    return "LOW"


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

ORGANISM_ALIASES = {
    "s. typhimurium":       ["salmonella typhimurium", "s. typhimurium"],
    "s. enteritidis":       ["salmonella enteritidis", "s. enteritidis"],
    "s. infantis":          ["salmonella infantis", "s. infantis"],
    "s. kentucky":          ["salmonella kentucky", "s. kentucky"],
    "s. derby":             ["salmonella derby", "s. derby"],
    "s. typhimurium (mono)":["salmonella typhimurium monophasic", "s. typhimurium (mono)"],
    "c. jejuni":            ["campylobacter jejuni", "campylobacter", "c. jejuni", "c. coli"],
    "c. coli":              ["campylobacter coli", "c. coli"],
    "e. coli":              ["escherichia coli", "e. coli"],
    "e. coli (esbl/ampc)":  ["escherichia coli esbl", "e. coli (esbl/ampc)", "e. coli esbl", "e. coli"],
    "e. faecalis":          ["enterococcus faecalis", "e. faecalis"],
    "e. faecium":           ["enterococcus faecium", "e. faecium"],
    "salmonella spp.":      ["salmonella spp.", "salmonella spp", "salmonella"],
}

def _organism_matches(actual: str, expected: str) -> bool:
    """Fuzzy organism matching — accepts full names and abbreviations."""
    actual_l   = actual.lower().strip()
    expected_l = expected.lower().strip()
    if actual_l == expected_l:
        return True
    aliases = ORGANISM_ALIASES.get(expected_l, [])
    return actual_l in aliases


def _antibiotic_matches(actual: str, expected: str) -> bool:
    actual_l   = actual.lower().strip()
    expected_l = expected.lower().strip()
    if actual_l == expected_l:
        return True
    aliases = ANTIBIOTIC_ALIASES.get(expected_l, [])
    return actual_l in aliases

ANTIBIOTIC_ALIASES = {
    "van": ["vancomycin", "van"],
    "cip": ["ciprofloxacin", "cip", "fluoroquinolone"],
    "amp": ["ampicillin", "amoxicillin", "amp"],
    "tet": ["tetracycline", "tet"],
    "gen": ["gentamicin", "gen"],
    "col": ["colistin", "col"],
    "esbl_pheno": ["esbl", "esbl_pheno", "extended spectrum"],
    "nal": ["nalidixic acid", "nalidixic", "nal"],
}

def _antibiotic_matches(actual: str, expected: str) -> bool:
    actual_l   = actual.lower().strip()
    expected_l = expected.lower().strip()
    if actual_l == expected_l:
        return True
    aliases = ANTIBIOTIC_ALIASES.get(expected_l, [])
    return actual_l in aliases

def eval_intent_and_entities(result: dict, expected: dict) -> dict:
    parsed = result.get("parsed_params", {})

    intent_ok = result.get("intent") == expected["expected_intent"]

    org_ok = True
    if expected["expected_organism"] is not None:
        actual_org = (parsed.get("organism") or "").strip()
        exp_org    = expected["expected_organism"].strip()
        org_ok     = _organism_matches(actual_org, exp_org)

    ab_ok = True
    if expected["expected_antibiotic"] is not None:
        actual_ab = (parsed.get("antibiotic") or "").strip()
        exp_ab    = expected["expected_antibiotic"].strip()
        ab_ok     = _antibiotic_matches(actual_ab, exp_ab)

    country_ok = True
    if expected["expected_country"] is not None:
        actual_c = (parsed.get("country") or "").strip().lower()
        exp_c    = expected["expected_country"].strip().lower()
        country_ok = actual_c == exp_c

    return {
        "intent_correct":     intent_ok,
        "organism_correct":   org_ok,
        "antibiotic_correct": ab_ok,
        "country_correct":    country_ok,
        "actual_intent":      result.get("intent"),
        "actual_organism":    parsed.get("organism"),
        "actual_antibiotic":  parsed.get("antibiotic"),
        "actual_country":     parsed.get("country"),
    }


def eval_citation_hallucination(result: dict) -> dict:
    answer   = result.get("answer", "")
    lit_hits = result.get("literature", {}).get("hits", [])

    actual_pmids = {str(h["pmid"]) for h in lit_hits}
    # Require 7-8 digit standalone PMIDs — avoids false positives from
    # numbers like "41" being matched inside "41963065"
    cited_pmids  = set(re.findall(r'PMID[:\s]+(\d{7,8})\b', answer, re.IGNORECASE))

    hallucinated = cited_pmids - actual_pmids

    return {
        "cited_pmids":        list(cited_pmids),
        "actual_pmids":       list(actual_pmids),
        "hallucinated_pmids": list(hallucinated),
        "hallucination_count":len(hallucinated),
        "clean":              len(hallucinated) == 0,
        "has_citations":      len(cited_pmids) > 0,
    }


def eval_severity_labels(result: dict) -> dict:
    answer = result.get("answer", "").upper()
    rows   = result.get("data", {}).get("rows", [])

    if not rows:
        return {"skipped": True, "reason": "no data rows returned"}

    # Extract percentages actually mentioned in the answer
    mentioned_pcts = set()
    import re
    for m in re.findall(r'(\d+\.?\d*)\s*%', result.get("answer", "")):
        mentioned_pcts.add(round(float(m), 1))

    checks = []
    for row in rows:
        pct = row.get("avg_pct_resistant") or row.get("percent_resistant")
        if pct is None:
            continue
        pct     = float(pct)
        pct_r   = round(pct, 1)
        # Only check rows whose percentage actually appears in the answer
        if pct_r not in mentioned_pcts:
            continue
        exp_lbl = expected_severity(pct)
        found   = exp_lbl in answer
        present = [lbl for _, lbl in SEVERITY_THRESHOLDS if lbl in answer]
        checks.append({
            "pct":              pct_r,
            "expected_label":   exp_lbl,
            "label_found":      found,
            "labels_in_answer": present,
        })

    if not checks:
        return {"skipped": True, "reason": "no data percentages mentioned in answer"}

    correct = sum(1 for c in checks if c["label_found"])
    return {
        "skipped":  False,
        "checks":   checks,
        "correct":  correct,
        "total":    len(checks),
        "accuracy": round(correct / len(checks), 3),
    }

def eval_answer_quality(question: str, answer: str, data_rows: list) -> dict:
    """
    LLM-as-judge: scores answer quality 0-3 on three axes (9 total).
    Uses the model defined in config (NARRATIVE_MODEL) via the configured API.
    Only called when at least one rule-based check has already failed.
    """
    try:
        from config import API_KEY, API_BASE_URL, NARRATIVE_MODEL

        data_summary = json.dumps(data_rows[:5], indent=2) if data_rows else "No data returned"
        answer_trunc = answer[:800] + ("…" if len(answer) > 800 else "")

        prompt = f"""You are evaluating an AMR (antimicrobial resistance) surveillance system.

Question: {question}

Data provided to the system (first 5 rows):
{data_summary}

System answer (truncated to 800 chars):
{answer_trunc}

Rate on exactly these three criteria. Reply ONLY with valid JSON, no markdown fences.

{{
  "factual_grounding": {{"score": 0, "comment": "one sentence"}},
  "completeness":      {{"score": 0, "comment": "one sentence"}},
  "clinical_clarity":  {{"score": 0, "comment": "one sentence"}}
}}

Scoring: 3=fully correct, 2=mostly correct/minor gap, 1=partial/missing key info, 0=wrong/misleading
factual_grounding: Does the answer only use data actually in the provided rows?
completeness: Does it fully address the question?
clinical_clarity: Is it clear and clinically appropriate?"""

        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model":   NARRATIVE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {"num_predict": 300},
            },
            timeout=120,
        )
        response.raise_for_status()
        raw    = response.json()["message"]["content"].strip()
        raw    = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)
        total  = sum(v["score"] for v in scores.values())
        return {
            "skipped": False,
            "scores":  scores,
            "total":   total,
            "pct":     round(total / 9, 3),
        }
    except Exception as e:
        return {"skipped": True, "reason": str(e)}


# ============================================================
# PRETTY PRINTERS
# ============================================================

def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

def _pct(value: Optional[float]) -> str:
    return "  N/A " if value is None else f"{value*100:5.1f}%"


def print_report(
    total:           int,
    intent_correct:  int,
    entity_correct:  int,
    entity_total:    int,
    hall_clean:      int,
    hall_total:      int,
    sev_correct:     int,
    sev_total:       int,
    llm_score_total: int,
    llm_score_count: int,
    failures:        list,
) -> dict:

    intent_acc = intent_correct / total
    entity_acc = entity_correct / entity_total if entity_total > 0 else None
    hall_acc   = hall_clean     / hall_total   if hall_total   > 0 else None
    sev_acc    = sev_correct    / sev_total    if sev_total    > 0 else None
    llm_acc    = llm_score_total / llm_score_count if llm_score_count > 0 else None

    scores  = [s for s in [intent_acc, entity_acc, hall_acc, sev_acc, llm_acc] if s is not None]
    overall = sum(scores) / len(scores) if scores else 0.0

    W = 62
    print(f"\n{'═'*W}")
    print(f"  AMR SENTINEL — EVALUATION REPORT  ({total} questions)")
    print(f"{'═'*W}")
    print(f"\n  {'Metric':<35} {'Score':>8}   {'Bar'}")
    print(f"  {'─'*35}   {'─'*8}   {'─'*30}")

    rows_m = [
        ("1. Intent Accuracy",
         f"{intent_correct}/{total}", intent_acc),
        ("2. Entity Accuracy (org/drug/country)",
         f"{entity_correct}/{entity_total}" if entity_total else "N/A", entity_acc),
        ("3. No-Hallucination Rate",
         f"{hall_clean}/{hall_total}", hall_acc),
        ("4. Severity Label Accuracy",
         f"{sev_correct}/{sev_total}" if sev_total else "N/A", sev_acc),
        ("5. Answer Quality (LLM judge)*",
         f"{llm_score_total}/{llm_score_count}" if llm_score_count else "N/A", llm_acc),
    ]

    for label, fraction, acc in rows_m:
        bar = _bar(acc) if acc is not None else " " * 32
        print(f"  {label:<35} {fraction:>8}   {bar}  {_pct(acc)}")

    print(f"\n  {'─'*(W-2)}")
    print(f"  {'Overall Score':<35} {'':>8}   {_bar(overall)}  {_pct(overall)}")
    print(f"{'═'*W}")
    print(f"  * LLM judge runs on all questions (gemma3:27b via Ollama)\n")

    if failures:
        print(f"  FAILURES ({len(failures)} question(s) had at least one issue)\n")
        for f in failures:
            print(f"  [{f['idx']:02d}] {f['question'][:65]}")
            for issue in f["issues"]:
                print(f"       ✗  {issue}")
            print()
    else:
        print("  ✓ No failures — all checks passed!\n")

    print(f"{'═'*W}\n")

    return {
        "total_questions":          total,
        "intent_accuracy":          round(intent_acc, 3),
        "entity_accuracy":          round(entity_acc, 3) if entity_acc is not None else None,
        "hallucination_clean_rate": round(hall_acc,   3) if hall_acc   is not None else None,
        "severity_accuracy":        round(sev_acc,    3) if sev_acc    is not None else None,
        "llm_quality_score":        round(llm_acc,    3) if llm_acc    is not None else None,
        "overall_score":            round(overall,    3),
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def run_evaluation(verbose: bool = False) -> dict:
    total = len(EVAL_QUESTIONS)
    print(f"\n  Running {total} evaluation questions …  (this may take a minute)\n")

    intent_correct  = 0
    entity_correct  = 0
    entity_total    = 0
    hall_clean      = 0
    hall_total      = 0
    sev_correct     = 0
    sev_total       = 0
    llm_score_total = 0
    llm_score_count = 0

    all_results = []
    failures    = []

    for i, eq in enumerate(EVAL_QUESTIONS, 1):
        q       = eq["question"]
        short_q = q[:65] + ("…" if len(q) > 65 else "")
        print(f"  [{i:02d}/{total}] {short_q}", flush=True)

        try:
            result = ask(q)
        except Exception as e:
            print(f"           ERROR: {e}")
            failures.append({"idx": i, "question": q, "issues": [f"Exception: {e}"]})
            continue

        issues = []

        # ── 1. Intent + entities ──────────────────────────────
        ie = eval_intent_and_entities(result, eq)

        if ie["intent_correct"]:
            intent_correct += 1
        else:
            issues.append(
                f"Intent: expected '{eq['expected_intent']}', "
                f"got '{ie['actual_intent']}'"
            )

        for field, exp_key in [
            ("organism_correct",   "expected_organism"),
            ("antibiotic_correct", "expected_antibiotic"),
            ("country_correct",    "expected_country"),
        ]:
            if eq.get(exp_key) is not None:
                entity_total += 1
                if ie[field]:
                    entity_correct += 1
                else:
                    issues.append(
                        f"{exp_key.replace('expected_', '').title()}: "
                        f"expected '{eq[exp_key]}', "
                        f"got '{ie.get('actual_' + exp_key.replace('expected_', ''))}'"
                    )

        # ── 2. Citation hallucination ─────────────────────────
        ch = eval_citation_hallucination(result)
        hall_total += 1
        if ch["clean"]:
            hall_clean += 1
        else:
            issues.append(
                f"Hallucinated PMIDs: {ch['hallucinated_pmids']} "
                f"(cited {ch['cited_pmids']}, retrieved {ch['actual_pmids']})"
            )

        # ── 3. Severity labels ────────────────────────────────
        sl = eval_severity_labels(result)
        if not sl.get("skipped"):
            sev_correct += sl["correct"]
            sev_total   += sl["total"]
            if sl["accuracy"] < 1.0:
                for w in [c for c in sl["checks"] if not c["label_found"]][:3]:
                    issues.append(
                        f"Severity: {w['pct']}% → expected '{w['expected_label']}' "
                        f"but absent (answer has {w['labels_in_answer']})"
                    )

        # ── 4. LLM judge — runs on all questions ────────────
        aq = eval_answer_quality(
            q,
            result.get("answer", ""),
            result.get("data", {}).get("rows", [])
        )

        if not aq.get("skipped"):
            llm_score_total += aq["total"]
            llm_score_count += 9
            if aq["pct"] < 0.75:
                for criterion, v in aq["scores"].items():
                    if v["score"] < 2:
                        issues.append(
                            f"Quality/{criterion}: {v['score']}/3 — {v['comment']}"
                        )

        # ── Per-question one-liner ────────────────────────────
        q_mark = "✓" if ie["intent_correct"] else "✗"
        h_mark = "✓" if ch["clean"] else "✗"
        s_mark = ("–" if sl.get("skipped")
                  else ("✓" if sl.get("accuracy", 0) == 1.0 else "✗"))
        j_mark = ("–" if aq.get("skipped")
                  else ("✓" if aq.get("pct", 0) >= 0.75 else "✗"))

        print(f"           Intent {q_mark}  │  No-hallucination {h_mark}  │  "
              f"Severity {s_mark}  │  Quality {j_mark}")

        if verbose:
            print(f"           Intent:     exp={eq['expected_intent']}  "
                  f"got={ie['actual_intent']}")
            print(f"           Organism:   exp={eq['expected_organism']}  "
                  f"got={ie['actual_organism']}")
            print(f"           Antibiotic: exp={eq['expected_antibiotic']}  "
                  f"got={ie['actual_antibiotic']}")
            if ch["hallucinated_pmids"]:
                print(f"           HALLUCINATED PMIDs: {ch['hallucinated_pmids']}")
            print(f"           Answer[:300]: {result.get('answer', '')[:300]}\n")

        if issues:
            failures.append({"idx": i, "question": q, "issues": issues})

        all_results.append({
            "question":       q,
            "intent_entity":  ie,
            "citation":       ch,
            "severity":       sl,
            "answer_quality": aq,
            "data_row_count": result.get("data", {}).get("row_count", 0),
            "lit_hit_count":  result.get("literature", {}).get("hit_count", 0),
        })

    # ── Final report ─────────────────────────────────────────
    summary = print_report(
        total, intent_correct,
        entity_correct, entity_total,
        hall_clean, hall_total,
        sev_correct, sev_total,
        llm_score_total, llm_score_count,
        failures,
    )
    summary["details"] = all_results
    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMR Sentinel Factual Evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print answer excerpts")
    parser.add_argument("--output",  type=str,            help="Save JSON report to file")
    args = parser.parse_args()

    report = run_evaluation(verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved → {args.output}\n")
