import json
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from datetime import datetime

from orchestrator import ask as original_rag
from baseline_rag import single_agent_rag
from config import API_KEY, API_BASE_URL, NARRATIVE_MODEL

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

JUDGE_MODEL = "gemini-3.1-flash-lite"



def normalize_score(value):
    try:
        score = float(value)
        return max(0.0, min(3.0, score))
    except (TypeError, ValueError):
        return 0.0

# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    "How has vancomycin resistance in Enterococcus faecium changed over time in Europe?",
    "Show me the trend of ampicillin resistance in Klebsiella pneumoniae over the years",
    "Has colistin resistance in E. coli increased or decreased over time?",
    "How has meropenem resistance in Acinetobacter baumannii changed in Italy?",
    "Show me the trend of penicillin resistance in Streptococcus pneumoniae across Europe",
    "How has tetracycline resistance in E. coli in pig farms changed over time?",
    "How has ciprofloxacin resistance in E. coli changed over time?",
    "Show me the trend of tetracycline resistance in Salmonella Typhimurium",
    "How has ampicillin resistance in E. coli changed in Germany over the years?",
    "Is fluoroquinolone resistance in Campylobacter increasing?",
    "Which European countries have the highest ampicillin resistance in Salmonella Enteritidis?",
    "Compare meropenem resistance in Klebsiella pneumoniae across European countries",
    "Which countries have the lowest ciprofloxacin resistance in E. coli in broilers?",
    "Compare colistin resistance in E. coli between pig farms and broiler farms",
    "Which countries have the highest gentamicin resistance in E. coli in pork meat?",
    "Compare vancomycin resistance in Enterococcus faecalis across Europe",
    "Compare ciprofloxacin resistance in E. coli across European countries",
    "Which countries have the highest tetracycline resistance in Salmonella?",
    "Compare vancomycin resistance in Enterococcus faecium across countries",
    "Compare E. coli resistance to tetracycline between pig farms and human bloodstream infections across Europe",
    "What are the most resistant organism and antibiotic combinations in animal sources?",
    "Show me the top 5 highest resistance combinations in human bloodstream infections",
    "Which antibiotic has the highest overall resistance rate across all organisms?",
    "What are the worst resistance combinations in broiler meat across Europe?",
    "Which organism and antibiotic combinations have the highest resistance?",
    "Show me the top 10 highest resistance combinations",
    "Which antibiotics are tracked for Salmonella in the database?",
    "What years of data are available for human bloodstream infections?",
    "Which animal source types are covered in the system?",
    "What Salmonella serovars are included in the zoonotic data?",
    "What data is available in this system?",
    "Which countries are included in the EFSA zoonotic dataset?"
]

# ============================================================
# LLM CALL
# ============================================================

def call_judge(prompt):
    """Call Gemini with retry handling.
    IMPORTANT SCORING RULE:

The three scores MUST be integers.

Valid values are ONLY:
0, 1, 2, or 3.

NEVER return:
- percentages
- decimals
- values such as 60, 75, 85, 90, or 100

For example, if an answer is very well grounded, return:
"factual_grounding_score": 3

NOT:
"factual_grounding_score": 100"""

    import time

    max_retries = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt
            )

            return response.text

        except Exception as e:
            print(f"Gemini error (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                raise


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    raise ValueError(
        f"Could not parse judge JSON:\n{text[:2000]}"
    )


# ============================================================
# RUN ORIGINAL RAG
# ============================================================

def run_original(question):

    print("\nRunning ORIGINAL RAG...")
    print(question)

    try:
        result = original_rag(question)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "evidence": result,
        }

    except Exception as e:

        return {
            "question": question,
            "answer": "",
            "evidence": {},
            "error": str(e),
        }


# ============================================================
# RUN SINGLE-AGENT BASELINE
# ============================================================

def run_baseline(question):

    print("\nRunning SINGLE-AGENT RAG...")
    print(question)

    try:
        result = single_agent_rag(question)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "evidence": result,
        }

    except Exception as e:

        return {
            "question": question,
            "answer": "",
            "evidence": {},
            "error": str(e),
        }


# ============================================================
# NEW JUDGE PROMPT
# ============================================================

def build_single_judge_prompt(question, answer, evidence):

    evidence_text = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        default=str
    )

    return f"""
You are evaluating an antimicrobial resistance RAG system.

Your task is to determine how well the answer is grounded in the
retrieved evidence.

IMPORTANT:

Use ONLY the evidence supplied below.

Do NOT use your own medical knowledge.
Do NOT assume a statement is true because it sounds reasonable.

A hallucination means that a CORE FACTUAL CLAIM in the answer is:

1. NOT supported by the supplied evidence, OR
2. CONTRADICTED by the supplied evidence.

Do NOT automatically classify recommendations, advice, or general
interpretations as hallucinations.

------------------------------------------------------------
CORE FACTUAL CLAIMS
------------------------------------------------------------

Examples:

- resistance percentages
- numbers of isolates
- countries
- organisms
- antibiotics
- years
- trends directly claimed from the data
- rankings
- comparisons
- findings attributed to PubMed literature
- dataset characteristics

These MUST be grounded in the supplied evidence.

------------------------------------------------------------
RECOMMENDATIONS / INTERPRETATIONS
------------------------------------------------------------

Examples:

- "these antibiotics should not be used empirically"
- "susceptibility testing is recommended"
- "surveillance should be strengthened"
- clinical or policy recommendations
- general interpretation of what the numbers mean

Evaluate these separately.

They should NOT be included in the CORE hallucination rate.

------------------------------------------------------------
IMPORTANT DISTINCTION
------------------------------------------------------------

If the evidence does not contain enough information to verify a claim,
mark it as unsupported.

If the evidence explicitly conflicts with the claim,
mark it as contradicted.

If a statement is directly supported by the evidence,
mark it as supported.

Do not invent evidence.

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------

{answer}

------------------------------------------------------------
RETRIEVED EVIDENCE
------------------------------------------------------------

{evidence_text}

------------------------------------------------------------
RETURN ONLY VALID JSON
------------------------------------------------------------

Use exactly this structure:

{{
  "core_factual_claims": [
    {{
      "claim": "short factual claim",
      "status": "supported",
      "evidence": "brief evidence supporting the claim"
    }}
  ],

  "recommendations_or_interpretations": [
    {{
      "claim": "recommendation or interpretation",
      "supported_by_evidence": true,
      "evidence": "brief explanation"
    }}
  ],

  "factual_grounding_score": 0,
  "completeness_score": 0,
  "clarity_score": 0,

  "overall_assessment": "brief assessment"
}}

For core_factual_claims, status MUST be one of:

- "supported"
- "unsupported"
- "contradicted"

Scores:

factual_grounding_score:
0 = mostly unsupported or contradicted
1 = partially grounded
2 = mostly grounded
3 = very well grounded

completeness_score:
0 = does not answer the question
1 = partially answers it
2 = mostly answers it
3 = completely answers it

clarity_score:
0 = unclear
1 = somewhat clear
2 = clear
3 = very clear
"""


# ============================================================
# JUDGE SINGLE SYSTEM
# ============================================================

def judge_single_system(question, result):

    prompt = build_single_judge_prompt(
        question,
        result["answer"],
        result["evidence"]
    )

    raw = call_judge(prompt)

    try:
        judged = extract_json(raw)
    except Exception as e:

        print("WARNING: Judge JSON parsing failed.")

        judged = {
            "core_factual_claims": [],
            "recommendations_or_interpretations": [],
            "factual_grounding_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "overall_assessment": raw,
            "judge_error": str(e),
        }

    calculate_claim_metrics(judged)

    return judged


# ============================================================
# CLAIM METRICS
# ============================================================

def calculate_claim_metrics(judged):

    claims = judged.get("core_factual_claims", [])

    supported = 0
    unsupported = 0
    contradicted = 0

    for claim in claims:

        status = str(
            claim.get("status", "")
        ).lower().strip()

        if status == "supported":
            supported += 1

        elif status == "unsupported":
            unsupported += 1

        elif status == "contradicted":
            contradicted += 1

    total = supported + unsupported + contradicted

    hallucinations = unsupported + contradicted

    if total > 0:
        hallucination_rate = hallucinations / total
        grounding_rate = supported / total
    else:
        hallucination_rate = 0
        grounding_rate = 0

    judged["metrics"] = {
        "total_core_factual_claims": total,
        "supported_core_claims": supported,
        "unsupported_core_claims": unsupported,
        "contradicted_core_claims": contradicted,
        "core_hallucinations": hallucinations,
        "core_hallucination_rate": hallucination_rate,
        "core_grounding_rate": grounding_rate,
    }

    # Recommendations are reported separately
    recommendations = judged.get(
        "recommendations_or_interpretations",
        []
    )

    rec_total = len(recommendations)

    rec_supported = sum(
        1
        for r in recommendations
        if r.get("supported_by_evidence") is True
    )

    rec_unsupported = rec_total - rec_supported

    if rec_total > 0:
        rec_rate = rec_unsupported / rec_total
    else:
        rec_rate = 0

    judged["metrics"]["recommendations"] = {
        "total": rec_total,
        "supported": rec_supported,
        "unsupported": rec_unsupported,
        "unsupported_rate": rec_rate,
    }


# ============================================================
# COMPARISON JUDGE
# ============================================================

def build_comparison_prompt(
    question,
    original_result,
    baseline_result
):

    original_evidence = json.dumps(
        original_result["evidence"],
        indent=2,
        ensure_ascii=False,
        default=str
    )

    baseline_evidence = json.dumps(
        baseline_result["evidence"],
        indent=2,
        ensure_ascii=False,
        default=str
    )

    return f"""
You are comparing TWO antimicrobial resistance RAG systems.

Evaluate both answers ONLY against the evidence supplied for each
system.

Do NOT use outside medical knowledge.

Question:

{question}

============================================================
ORIGINAL RAG ANSWER
============================================================

{original_result["answer"]}

ORIGINAL RAG EVIDENCE:

{original_evidence}

============================================================
SINGLE-AGENT RAG ANSWER
============================================================

{baseline_result["answer"]}

SINGLE-AGENT RAG EVIDENCE:

{baseline_evidence}

============================================================
EVALUATION RULES
============================================================

For each answer identify CORE FACTUAL CLAIMS.

Core factual claims include:

- resistance percentages
- isolate counts
- organisms
- antibiotics
- countries
- years
- trends
- rankings
- comparisons
- literature findings
- dataset-derived statistics

A core claim is:

SUPPORTED
if directly supported by the system's supplied evidence.

UNSUPPORTED
if the evidence does not support it.

CONTRADICTED
if the evidence conflicts with it.

Recommendations and interpretations must be evaluated separately.
Do NOT count them as core hallucinations.

============================================================
RETURN ONLY VALID JSON
============================================================

{{
  "original": {{
    "core_factual_claims": [
      {{
        "claim": "...",
        "status": "supported",
        "evidence": "..."
      }}
    ],
    "recommendations_or_interpretations": [
      {{
        "claim": "...",
        "supported_by_evidence": true
      }}
    ],
    "factual_grounding_score": 0,
    "completeness_score": 0,
    "clarity_score": 0
  }},

  "single_agent": {{
    "core_factual_claims": [
      {{
        "claim": "...",
        "status": "supported",
        "evidence": "..."
      }}
    ],
    "recommendations_or_interpretations": [
      {{
        "claim": "...",
        "supported_by_evidence": true
      }}
    ],
    "factual_grounding_score": 0,
    "completeness_score": 0,
    "clarity_score": 0
  }},

  "winner": "original",
  "reason": "brief explanation"
}}

winner MUST be:

- "original"
- "single_agent"
- "tie"

Do not choose a winner based only on answer length.
Prioritize factual grounding and completeness.
"""


# ============================================================
# NORMALIZE COMPARISON RESULT
# ============================================================

def normalize_comparison(judged):

    for system in ["original", "single_agent"]:

        if system not in judged:
            continue

        calculate_claim_metrics(
            judged[system]
        )

    return judged


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(results):

    total_claims = 0
    total_hallucinations = 0
    total_score = 0
    score_count = 0

    for item in results:

        metrics = item["judge"]["metrics"]

        total_claims += metrics[
            "total_core_factual_claims"
        ]

        total_hallucinations += metrics[
            "core_hallucinations"
        ]

        judge = item["judge"]

        total_score += (
            normalize_score(judge.get("factual_grounding_score", 0))
            + normalize_score(judge.get("completeness_score", 0))
            + normalize_score(judge.get("clarity_score", 0))
        )

        score_count += 1

    if total_claims > 0:
        hallucination_rate = (
            total_hallucinations / total_claims
        )
    else:
        hallucination_rate = 0

    if score_count > 0:
        average_score = total_score / score_count
    else:
        average_score = 0

    return {
        "questions": len(results),
        "total_core_factual_claims": total_claims,
        "total_core_hallucinations": total_hallucinations,
        "core_hallucination_rate": hallucination_rate,
        "average_score_out_of_9": average_score,
    }


# ============================================================
# COMPARISON SUMMARY
# ============================================================

def calculate_comparison_summary(results):

    original_claims = 0
    original_hallucinations = 0
    original_score = 0

    baseline_claims = 0
    baseline_hallucinations = 0
    baseline_score = 0

    original_wins = 0
    baseline_wins = 0
    ties = 0

    for item in results:

        original = item["judge"]["original"]
        baseline = item["judge"]["single_agent"]

        om = original["metrics"]
        bm = baseline["metrics"]

        original_claims += om[
            "total_core_factual_claims"
        ]

        original_hallucinations += om[
            "core_hallucinations"
        ]

        baseline_claims += bm[
            "total_core_factual_claims"
        ]

        baseline_hallucinations += bm[
            "core_hallucinations"
        ]

        original_score += (
            normalize_score(original.get("factual_grounding_score", 0))
            + normalize_score(original.get("completeness_score", 0))
            + normalize_score(original.get("clarity_score", 0))
        )

        baseline_score += (
            normalize_score(baseline.get("factual_grounding_score", 0))
            + normalize_score(baseline.get("completeness_score", 0))
            + normalize_score(baseline.get("clarity_score", 0))
        )

        winner = item["judge"].get(
            "winner",
            "tie"
        )

        if winner == "original":
            original_wins += 1

        elif winner == "single_agent":
            baseline_wins += 1

        else:
            ties += 1

    original_rate = (
        original_hallucinations / original_claims
        if original_claims
        else 0
    )

    baseline_rate = (
        baseline_hallucinations / baseline_claims
        if baseline_claims
        else 0
    )

    # Percentage reduction in hallucination rate
    if original_rate > 0:
        reduction = ((baseline_rate - original_rate) / baseline_rate) * 100
    else:
        reduction = 0

    return {
        "original": {
            "total_core_factual_claims": original_claims,
            "core_hallucinations": original_hallucinations,
            "core_hallucination_rate": original_rate,
            "average_score_out_of_9": (
                original_score / len(results)
                if results
                else 0
            ),
        },

        "single_agent": {
            "total_core_factual_claims": baseline_claims,
            "core_hallucinations": baseline_hallucinations,
            "core_hallucination_rate": baseline_rate,
            "average_score_out_of_9": (
                baseline_score / len(results)
                if results
                else 0
            ),
        },

        "baseline_to_original_hallucination_reduction_percent": round(reduction, 2),

        "wins": {
            "original": original_wins,
            "single_agent": baseline_wins,
            "tie": ties,
        }
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_results(data, prefix):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{prefix}_{timestamp}.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    return filename


# ============================================================
# OPTION 1 — ORIGINAL ONLY
# ============================================================

def evaluate_original():

    results = []

    print("\n")
    print("=" * 70)
    print("ORIGINAL RAG EVALUATION")
    print("=" * 70)

    for i, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"\n[{i}/{len(QUESTIONS)}]"
        )

        result = run_original(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nJudging...")

        judge = judge_single_system(
            question,
            result
        )

        results.append({
            "question": question,
            "result": result,
            "judge": judge,
        })

        print(
            "Core hallucination rate:",
            f"{judge['metrics']['core_hallucination_rate']:.2%}"
        )

    summary = calculate_summary(results)

    output = {
        "evaluation_mode": "original_only",
        "judge_model": JUDGE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "questions": QUESTIONS,
        "results": results,
        "summary": summary,
    }

    filename = save_results(
        output,
        "evaluation_original"
    )

    print("\n")
    print("=" * 70)
    print("ORIGINAL RAG SUMMARY")
    print("=" * 70)

    print(
        f"Average score: "
        f"{summary['average_score_out_of_9']:.2f}/9"
    )

    print(
        f"Core factual claims: "
        f"{summary['total_core_factual_claims']}"
    )

    print(
        f"Core hallucinations: "
        f"{summary['total_core_hallucinations']}"
    )

    print(
        f"Core hallucination rate: "
        f"{summary['core_hallucination_rate']:.2%}"
    )

    print(
        f"\nResults saved to:\n{filename}"
    )

def evaluate_baseline():
    results = []

    print("\n" + "=" * 70)
    print("SINGLE-AGENT RAG EVALUATION")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] {question}")

        try:
            result = run_baseline(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nJudging...")
            judge = judge_single_system(question, result)

            results.append({
                "question": question,
                "result": result,
                "judge": judge,
            })

            print(
                "Core hallucination rate:",
                f"{judge['metrics']['core_hallucination_rate']:.2%}"
            )

        except Exception as e:
            print(f"ERROR: {e}")

    summary = calculate_summary(results)

    output = {
        "evaluation_mode": "single_agent_only",
        "judge_model": JUDGE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "questions": QUESTIONS,
        "results": results,
        "summary": summary,
    }

    filename = save_results(output, "evaluation_single_agent")

    print("\n" + "=" * 70)
    print("SINGLE-AGENT RAG SUMMARY")
    print("=" * 70)

    print(f"Core claims:        {summary['total_core_factual_claims']}")
    print(f"Hallucinations:     {summary['total_core_hallucinations']}")
    print(f"Hallucination rate: {summary['core_hallucination_rate']:.2%}")
    print(f"Average score:      {summary['average_score_out_of_9']:.2f}/9")

    print(f"\nSaved to: {filename}")

    return output


# ============================================================
# OPTION 3 — ORIGINAL VS BASELINE
# ============================================================

def evaluate_comparison():

    results = []

    print("\n")
    print("=" * 70)
    print("ORIGINAL RAG VS SINGLE-AGENT RAG")
    print("=" * 70)

    for i, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"\n[{i}/{len(QUESTIONS)}]"
        )

        original = run_original(question)

        baseline = run_baseline(question)

        print("\nJudging comparison...")

        prompt = build_comparison_prompt(
            question,
            original,
            baseline
        )

        raw = call_judge(prompt)

        try:
            judge = extract_json(raw)

        except Exception as e:

            print(
                "WARNING: comparison judge JSON failed."
            )

            judge = {
                "original": {},
                "single_agent": {},
                "winner": "tie",
                "reason": raw,
                "judge_error": str(e),
            }

        normalize_comparison(judge)

        results.append({
            "question": question,
            "original": original,
            "single_agent": baseline,
            "judge": judge,
        })

        print(
            "Winner:",
            judge.get("winner", "tie")
        )

    summary = calculate_comparison_summary(
        results
    )

    output = {
        "evaluation_mode": "comparison",
        "judge_model": JUDGE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "questions": QUESTIONS,
        "results": results,
        "summary": summary,
    }

    filename = save_results(
        output,
        "evaluation_comparison"
    )

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    original = summary["original"]
    baseline = summary["single_agent"]

    print("\nORIGINAL RAG")
    print(
        f"  Core hallucinations: "
        f"{original['core_hallucinations']}"
    )
    print(
        f"  Core claims: "
        f"{original['total_core_factual_claims']}"
    )
    print(
        f"  Hallucination rate: "
        f"{original['core_hallucination_rate']:.2%}"
    )
    print(
        f"  Average score: "
        f"{original['average_score_out_of_9']:.2f}/9"
    )

    print("\nSINGLE-AGENT RAG")
    print(
        f"  Core hallucinations: "
        f"{baseline['core_hallucinations']}"
    )
    print(
        f"  Core claims: "
        f"{baseline['total_core_factual_claims']}"
    )
    print(
        f"  Hallucination rate: "
        f"{baseline['core_hallucination_rate']:.2%}"
    )
    print(
        f"  Average score: "
        f"{baseline['average_score_out_of_9']:.2f}/9"
    )

    print("\nBASELINE HALLUCINATION RATE CHANGE VS ORIGINAL")

    print(
        f"  {summary['baseline_to_original_hallucination_reduction_percent']:.2f}%"
    )

    print("\nWINS")

    print(
        f"  Original RAG: "
        f"{summary['wins']['original']}"
    )

    print(
        f"  Single-Agent RAG: "
        f"{summary['wins']['single_agent']}"
    )

    print(
        f"  Ties: "
        f"{summary['wins']['tie']}"
    )

    print(
        f"\nResults saved to:\n{filename}"
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("\n" + "=" * 70)
print("AMR SENTINEL EVALUATION")
print("=" * 70)
print("1. Evaluate Original RAG only")
print("2. Evaluate Single-Agent RAG only")
print("3. Compare Original RAG vs Single-Agent RAG")

choice = input("\nEnter choice (1/2/3): ").strip()

if choice == "1":
    evaluate_original()

elif choice == "2":
    evaluate_baseline()

elif choice == "3":
    evaluate_comparison()

else:
    print("Invalid choice.")


if __name__ == "__main__":
    main()