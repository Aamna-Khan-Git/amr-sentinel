import json
import traceback

# Original RAG
from orchestrator import ask as original_rag

# Single-Agent RAG
from baseline_rag import single_agent_rag


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    "How has ciprofloxacin resistance in E. coli changed over time?",
    "Show me the trend of tetracycline resistance in Salmonella Typhimurium",
    "How has ampicillin resistance in E. coli changed in Germany over the years?",
    "Trend of ESBL resistance in E. coli over the years",
    "Compare ciprofloxacin resistance in E. coli across European countries",
    "Which countries have the highest tetracycline resistance in Salmonella?",
    "Compare vancomycin resistance in Enterococcus faecium across countries",
    "Which organism and antibiotic combinations have the highest resistance?",
    "Show me the top 10 highest resistance combinations",
    "Is fluoroquinolone resistance in Campylobacter increasing?"
]


# ============================================================
# RUN ONE SYSTEM
# ============================================================

def run_original(question):
    try:
        result = original_rag(question)

        return {
            "answer": result.get("answer", ""),
            "intent": result.get("intent"),
            "parsed_params": result.get("parsed_params"),
            "data": result.get("data"),
            "literature": result.get("literature"),
            "error": result.get("error")
        }

    except Exception as e:
        return {
            "answer": "",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_single_agent(question):
    try:
        result = single_agent_rag(question)

        return {
            "answer": result.get("answer", ""),
            "query_parameters": result.get("query_parameters"),
            "data": result.get("data"),
            "literature": result.get("literature"),
            "error": None
        }

    except Exception as e:
        return {
            "answer": "",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    results = []

    print("=" * 70)
    print("AMR SENTINEL - RAG COMPARISON")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, start=1):

        print(f"\n[{i}/{len(QUESTIONS)}]")
        print("Question:", question)

        # ----------------------------------------------------
        # Original RAG
        # ----------------------------------------------------

        print("\nRunning Original RAG...")

        original = run_original(question)

        if original["error"]:
            print("Original RAG ERROR:", original["error"])
        else:
            print("Original RAG completed.")

        # ----------------------------------------------------
        # Single-Agent RAG
        # ----------------------------------------------------

        print("Running Single-Agent RAG...")

        single_agent = run_single_agent(question)

        if single_agent["error"]:
            print("Single-Agent RAG ERROR:", single_agent["error"])
        else:
            print("Single-Agent RAG completed.")

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({
            "question_id": i,
            "question": question,
            "original_rag": original,
            "single_agent_rag": single_agent
        })

    # ========================================================
    # SAVE JSON
    # ========================================================

    output_file = "rag_comparison_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nSaved results to: {output_file}")
    print(f"Questions evaluated: {len(results)}")


if __name__ == "__main__":
    main()