import json
import requests

from config import API_KEY, API_BASE_URL, NARRATIVE_MODEL


INPUT_FILE = "rag_comparison_results.json"
OUTPUT_FILE = "llm_judge_results.json"


# ============================================================
# LLM CALL
# ============================================================

def call_judge(prompt: str) -> str:

    if not API_KEY:
        raise RuntimeError("API_KEY is not configured.")

    base_url = API_BASE_URL.rstrip("/")

    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/api/chat"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NARRATIVE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a rigorous evaluator of antimicrobial resistance "
                    "RAG systems. Evaluate answers ONLY against the retrieved "
                    "evidence provided. Do not use outside medical knowledge."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    # OpenAI-compatible API
    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    # Generic API
    if "message" in data:
        return data["message"]["content"]

    raise RuntimeError(f"Unexpected API response: {data}")


# ============================================================
# JUDGE ONE QUESTION
# ============================================================

def judge_question(item):

    question = item["question"]

    original = item["original_rag"]
    single = item["single_agent_rag"]

    prompt = f"""
Evaluate the two RAG systems below.

IMPORTANT:
- Use ONLY the evidence provided for each system.
- Do NOT use outside knowledge.
- A factual statement is hallucinated if it is unsupported by the
  retrieved evidence or contradicts the retrieved evidence.
- Do not penalize an answer for information that the evidence itself
  does not contain.
- If the evidence is insufficient, saying so is considered good behavior.

============================================================
QUESTION
============================================================

{question}


============================================================
ORIGINAL RAG
============================================================

ANSWER:
{original.get("answer", "")}

STRUCTURED DATA:
{json.dumps(original.get("data", {}), indent=2, ensure_ascii=False)}

PUBMED LITERATURE:
{json.dumps(original.get("literature", []), indent=2, ensure_ascii=False)}


============================================================
SINGLE-AGENT RAG
============================================================

ANSWER:
{single.get("answer", "")}

STRUCTURED DATA:
{json.dumps(single.get("data", {}), indent=2, ensure_ascii=False)}

PUBMED LITERATURE:
{json.dumps(single.get("literature", []), indent=2, ensure_ascii=False)}


============================================================
SCORING
============================================================

For EACH system:

1. factual_grounding:
   0 = mostly unsupported or incorrect
   1 = partially grounded
   2 = mostly grounded with minor unsupported claims
   3 = fully grounded

2. completeness:
   0 = does not answer the question
   1 = partially answers it
   2 = mostly answers it
   3 = fully answers it

3. clinical_clarity:
   0 = misleading or unclear
   1 = weak interpretation
   2 = generally clear
   3 = clear, precise and appropriately cautious

4. hallucination_count:
   Count distinct factual claims in the answer that are unsupported
   or contradicted by the retrieved evidence.

5. hallucination_rate:
   Estimate:

   hallucinated factual claims / total factual claims

   Return this as a decimal between 0 and 1.

6. hallucination_examples:
   List the unsupported or contradicted claims.

Then compare the two systems.

Return ONLY valid JSON in exactly this structure:

{{
  "original_rag": {{
    "factual_grounding": 0,
    "completeness": 0,
    "clinical_clarity": 0,
    "total_score": 0,
    "hallucination_count": 0,
    "hallucination_rate": 0.0,
    "hallucination_examples": [],
    "reasoning": ""
  }},
  "single_agent_rag": {{
    "factual_grounding": 0,
    "completeness": 0,
    "clinical_clarity": 0,
    "total_score": 0,
    "hallucination_count": 0,
    "hallucination_rate": 0.0,
    "hallucination_examples": [],
    "reasoning": ""
  }},
  "winner": "original_rag | single_agent_rag | tie",
  "comparison_reason": ""
}}
"""

    raw = call_judge(prompt)

    # Remove accidental markdown fences
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "")
        raw = raw.replace("```", "")
        raw = raw.strip()

    return json.loads(raw)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMR SENTINEL - LLM AS A JUDGE")
    print("=" * 70)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        comparisons = json.load(f)

    results = []

    for i, item in enumerate(comparisons, start=1):

        print(f"\n[{i}/{len(comparisons)}]")
        print(item["question"])

        try:

            judgment = judge_question(item)

            results.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "judgment": judgment
            })

            print(
                "Winner:",
                judgment.get("winner")
            )

        except Exception as e:

            print("ERROR:", e)

            results.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "error": str(e)
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("JUDGING COMPLETE")
    print("=" * 70)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Questions judged: {len(results)}")


if __name__ == "__main__":
    main()