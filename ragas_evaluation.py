import json
from datetime import datetime

from orchestrator import ask as original_rag


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
    "Is fluoroquinolone resistance in Campylobacter increasing?",
]


def build_context(result):
    """Convert SQLite + PubMed evidence into RAGAS contexts."""

    contexts = []

    # Structured surveillance data
    data = result.get("data", {})

    if data:
        contexts.append(
            "STRUCTURED SURVEILLANCE DATA:\n"
            + json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

    # PubMed literature
    literature = result.get("literature", {})

    if literature:
        contexts.append(
            "PUBMED LITERATURE:\n"
            + json.dumps(
                literature,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

    return contexts


def run_evaluation():

    dataset = []

    print("=" * 70)
    print("RAGAS DATA COLLECTION — ORIGINAL RAG")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, start=1):

        print(f"\n[{i}/{len(QUESTIONS)}] {question}")

        try:
            result = original_rag(question)

            answer = result.get("answer", "")
            contexts = build_context(result)

            dataset.append({
                "question": question,
                "answer": answer,
                "contexts": contexts
            })

            print("✓ Completed")

        except Exception as e:

            print(f"✗ Error: {e}")

            dataset.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "error": str(e)
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"ragas_dataset_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    print("\n" + "=" * 70)
    print("DATASET CREATED")
    print("=" * 70)
    print(f"Saved to: {filename}")
    print(f"Questions: {len(dataset)}")


if __name__ == "__main__":
    run_evaluation()