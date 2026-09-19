import json
import glob
import os
import traceback
from datetime import datetime

from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
)

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# OLLAMA CLOUD API — RAGAS EVALUATOR
# ============================================================

# Ollama Cloud exposes an OpenAI-compatible /v1/chat/completions endpoint.
# Using ChatOpenAI here is intentional: it gives RAGAS the complete
# LangChain BaseChatModel interface it expects, including agenerate_prompt().
from config import API_KEY, API_BASE_URL
from langchain_openai import ChatOpenAI

EVALUATION_MODEL = "gemma4:31b-cloud"

if not API_KEY:
    raise ValueError(
        "API_KEY is not set in config.py/.env. "
        "Use the same Ollama Cloud API key used by orchestrator.py."
    )

if not API_BASE_URL:
    raise ValueError(
        "API_BASE_URL is not set in config.py/.env."
    )

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=EVALUATION_MODEL,
        temperature=0,
        max_tokens=6000,
        api_key=API_KEY,
        base_url=API_BASE_URL.rstrip("/"),
    )
)


# ============================================================
# EMBEDDINGS
# ============================================================

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

# ============================================================
# EXACT PROJECT FILES
# ============================================================

# These are the actual RAGAS input datasets in the project folder.
ORIGINAL_RAGAS_DATASET_FILE = "ragas_original_30.json"
SINGLE_AGENT_RAGAS_DATASET_FILE = "ragas_single_30.json"

# Limit context sent to the RAGAS judge to keep evaluation requests manageable.
MAX_CONTEXTS = 5
MAX_CONTEXT_CHARS = 2500



def require_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Required file not found: {filename}\n"
            f"Make sure run_ragas.py is being run from the project folder."
        )
    return filename


# ============================================================
# LOAD THE EXISTING RAGAS DATASET
# ============================================================

def load_ragas_dataset(dataset_file):
    """
    Load an existing RAGAS dataset.

    Both project datasets are already in RAGAS format:
    - question
    - answer
    - contexts

    or:
    - user_input
    - response
    - retrieved_contexts
    """

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # The files are JSON arrays.
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {dataset_file}, "
            f"but found {type(data).__name__}."
        )

    for item in data:
        # Original RAG dataset format
        if "question" in item:
            user_input = item.get("question", "")
            response = item.get("answer", "")
            contexts = item.get("contexts", [])

        # Single-Agent RAG dataset format
        elif "user_input" in item:
            user_input = item.get("user_input", "")
            response = item.get("response", "")
            contexts = item.get("retrieved_contexts", [])

        else:
            raise ValueError(
                f"Unrecognized RAGAS record format in {dataset_file}. "
                f"Available keys: {list(item.keys())}"
            )

        # RAGAS expects contexts to be a list of strings.
        if not isinstance(contexts, list):
            contexts = [str(contexts)]

        contexts = [str(c) for c in contexts if c is not None]

        # Keep the most relevant retrieved contexts and cap each context size.
        contexts = [c[:MAX_CONTEXT_CHARS] for c in contexts[:MAX_CONTEXTS]]

        records.append({
            "user_input": str(user_input),
            "retrieved_contexts": contexts,
            "response": str(response),
        })

    if not records:
        raise ValueError(f"No records found in {dataset_file}")

    return Dataset.from_list(records)


# ============================================================
# RUN RAGAS
# ============================================================

def evaluate_system(system_name, evaluation_file):
    print("\n" + "=" * 60)
    print(f"RAGAS — {system_name}")
    print("=" * 60)
    print(f"Using evaluation file: {evaluation_file}")

    dataset = load_ragas_dataset(evaluation_file)
    records = []

    print(f"Questions loaded: {len(dataset)}")
    print("\nStarting RAGAS evaluation...\n")

    run_config = RunConfig(
        max_workers=1,
        timeout=300,
        max_retries=5,
        max_wait=90,  # allow retries if the Ollama Cloud API is temporarily busy
    )

    # Evaluate one question at a time.
    # This prevents one failed metric generation from turning the
    # remaining samples into NaN and makes the exact failure visible.
    for i in range(len(dataset)):  # single-question smoke test; widen once this passes
        question = dataset[i]["user_input"]

        print("\n" + "-" * 60)
        print(f"Question {i + 1}/{len(dataset)}")
        print(question)

        single_dataset = Dataset.from_list([dataset[i]])

        try:
            result = evaluate(
                single_dataset,
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                ],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=True,
            )

            row = result.to_pandas().to_dict(orient="records")[0]

            # RAGAS 0.3.x can still return NaN in some edge cases even
            # when no Python exception escapes. Treat that as a failure
            # rather than as a real score.
            import math
            for metric_name in ("faithfulness", "answer_relevancy"):
                value = row.get(metric_name)
                if isinstance(value, float) and math.isnan(value):
                    raise RuntimeError(
                        f"RAGAS returned NaN for {metric_name} "
                        f"without exposing an exception."
                    )

            faithfulness = row.get("faithfulness")
            relevancy = row.get("answer_relevancy")

            print(f"Faithfulness: {faithfulness}")
            print(f"Answer Relevancy: {relevancy}")

            records.append(row)

        except Exception as e:
            print(f"RAGAS evaluation FAILED for Question {i + 1}:")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()

            # Keep the question in the output, but explicitly mark
            # the failed metrics as None rather than silently treating
            # the failure as a valid score.
            records.append({
                "user_input": dataset[i]["user_input"],
                "retrieved_contexts": dataset[i]["retrieved_contexts"],
                "response": dataset[i]["response"],
                "faithfulness": None,
                "answer_relevancy": None,
                "ragas_error": str(e),
            })

    # ------------------------------------------------------------
    # SAVE PER-QUESTION RESULTS
    # ------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = system_name.lower().replace(" ", "_").replace("-", "_")

    output_file = f"ragas_results_{safe_name}_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    # ------------------------------------------------------------
    # CALCULATE AVERAGES USING ONLY VALID SCORES
    # ------------------------------------------------------------

    faithfulness_scores = [
        r["faithfulness"]
        for r in records
        if isinstance(r.get("faithfulness"), (int, float))
        and r.get("faithfulness") == r.get("faithfulness")
    ]

    relevancy_scores = [
        r["answer_relevancy"]
        for r in records
        if isinstance(r.get("answer_relevancy"), (int, float))
        and r.get("answer_relevancy") == r.get("answer_relevancy")
    ]

    failed_questions = [
        i + 1
        for i, r in enumerate(records)
        if r.get("faithfulness") is None
        and r.get("answer_relevancy") is None
    ]

    summary = {
        "system": system_name,
        "evaluation_file": evaluation_file,
        "questions": len(records),
        "faithfulness_valid": len(faithfulness_scores),
        "answer_relevancy_valid": len(relevancy_scores),
        "failed_questions": failed_questions,
        "average_faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores else None
        ),
        "average_answer_relevancy": (
            sum(relevancy_scores) / len(relevancy_scores)
            if relevancy_scores else None
        ),
    }

    summary_file = f"ragas_summary_{safe_name}_{timestamp}.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RAGAS SUMMARY — {system_name}")
    print("=" * 60)

    if summary["average_faithfulness"] is not None:
        print(
            f"Average Faithfulness: "
            f"{summary['average_faithfulness']:.4f}"
        )
    else:
        print("Average Faithfulness: N/A")

    if summary["average_answer_relevancy"] is not None:
        print(
            f"Average Answer Relevancy: "
            f"{summary['average_answer_relevancy']:.4f}"
        )
    else:
        print("Average Answer Relevancy: N/A")

    print(
        f"Valid Faithfulness scores: "
        f"{summary['faithfulness_valid']}/{len(records)}"
    )
    print(
        f"Valid Answer Relevancy scores: "
        f"{summary['answer_relevancy_valid']}/{len(records)}"
    )

    if failed_questions:
        print(f"Failed questions: {failed_questions}")

    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

    return {
        "records": records,
        "summary": summary,
        "results_file": output_file,
        "summary_file": summary_file,
    }


# ============================================================
# MAIN MENU
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION PIPELINE")
    print("=" * 60)
    print("1. Evaluate Original RAG")
    print("2. Evaluate Single-Agent RAG")
    print("3. Evaluate Both Systems")
    print("4. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        # Your current folder does not contain an
        # evaluation_original_*.json file. The existing original
        # RAGAS dataset is used directly.
        original_file = require_file(ORIGINAL_RAGAS_DATASET_FILE)
        evaluate_system("Original RAG", original_file)

    elif choice == "2":
        single_agent_file = require_file(SINGLE_AGENT_RAGAS_DATASET_FILE)
        evaluate_system("Single-Agent RAG", single_agent_file)

    elif choice == "3":
        original_file = require_file(ORIGINAL_RAGAS_DATASET_FILE)
        single_agent_file = require_file(SINGLE_AGENT_RAGAS_DATASET_FILE)

        evaluate_system("Original RAG", original_file)
        evaluate_system("Single-Agent RAG", single_agent_file)

    elif choice == "4":
        print("Exiting.")
        return

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()