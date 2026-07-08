# evaluation/ragas_eval.py
# Dimension 08 — Runs RAGAS programmatically (not via LLM judge) on the
# dimension-08 test cases: faithfulness, answer relevancy, context precision,
# context recall. Uses the ragas library with retrieved_chunks as contexts.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rag_core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_RESULTS_PATH = _repo_root / "test_results.json"

# ---------------------------------------------------------------------------
# RAGAS configuration
# ---------------------------------------------------------------------------

# We need to set the OpenAI API key for RAGAS to use for its LLM-based metrics
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL


def compute_ragas_scores() -> dict:
    """Compute RAGAS metrics on dimension-08 test cases.

    Returns a dict with keys: faithfulness, answer_relevancy,
    context_precision, context_recall, plus a diagnosis string.
    """
    if not TEST_RESULTS_PATH.exists():
        print(f"[ragas_eval] ERROR: {TEST_RESULTS_PATH} not found. Run run_tests.py first.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "diagnosis": "No test results found — run run_tests.py first.",
        }

    with open(TEST_RESULTS_PATH, "r", encoding="utf-8") as f:
        test_results = json.load(f)

    # Filter to dimension-08 (ragas) test cases
    ragas_cases = [r for r in test_results if r.get("dimension", "").lower() == "ragas"]

    if not ragas_cases:
        print("[ragas_eval] WARNING: No dimension-08 (ragas) test cases found.")
        print("[ragas_eval] Will attempt to use all available test cases with retrieved_chunks.")
        ragas_cases = [r for r in test_results if r.get("retrieved_chunks")]

    if not ragas_cases:
        print("[ragas_eval] ERROR: No test cases with retrieved_chunks found.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "diagnosis": "No test cases with retrieved chunks available.",
        }

    print(f"[ragas_eval] Computing RAGAS metrics on {len(ragas_cases)} test cases …")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        print(f"[ragas_eval] ERROR: Missing dependency: {e}")
        print("[ragas_eval] Run: pip install ragas datasets")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "diagnosis": f"Missing dependency: {e}",
        }

    # Prepare data for RAGAS
    # RAGAS expects:
    #   - question: the user question
    #   - answer: the generated answer
    #   - contexts: list of retrieved context strings
    #   - ground_truth: the expected/reference answer
    data_rows = []
    for tc in ragas_cases:
        question = tc.get("question", "")
        answer = tc.get("actual_response", "")
        expected = tc.get("expected_answer", "")

        # Extract context texts from retrieved_chunks
        chunks = tc.get("retrieved_chunks", [])
        contexts = [c.get("text", "") for c in chunks if c.get("text")]

        if not contexts:
            # If no chunks, use a placeholder to avoid RAGAS errors
            contexts = ["No context retrieved"]

        data_rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": expected,
        })

    if not data_rows:
        print("[ragas_eval] ERROR: No valid data rows for RAGAS evaluation.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "diagnosis": "No valid data rows for RAGAS evaluation.",
        }

    # Create a HuggingFace Dataset
    dataset = Dataset.from_list(data_rows)

    print(f"[ragas_eval] Running RAGAS evaluation with {len(dataset)} samples …")
    print(f"[ragas_eval] Metrics: faithfulness, answer_relevancy, context_precision, context_recall")

    try:
        # Run RAGAS evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        # Convert to a serializable dict
        result_df = result.to_pandas()

        # Compute mean scores across all test cases
        scores = {
            "faithfulness": float(result_df["faithfulness"].mean()) if "faithfulness" in result_df.columns else 0.0,
            "answer_relevancy": float(result_df["answer_relevancy"].mean()) if "answer_relevancy" in result_df.columns else 0.0,
            "context_precision": float(result_df["context_precision"].mean()) if "context_precision" in result_df.columns else 0.0,
            "context_recall": float(result_df["context_recall"].mean()) if "context_recall" in result_df.columns else 0.0,
        }

        # Round to 2 decimal places
        scores = {k: round(v, 2) for k, v in scores.items()}

        # Determine diagnosis: which metric is lowest and why
        metric_names = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }

        lowest_metric = min(scores, key=scores.get)
        lowest_score = scores[lowest_metric]
        lowest_name = metric_names[lowest_metric]

        # Generate a concrete diagnosis
        if lowest_metric == "context_precision":
            diagnosis = (
                f"Context precision is lowest ({lowest_score:.2f}) — "
                f"the retrieved chunks contain irrelevant information. "
                f"Consider reducing chunk_size from 800 to 500, or adding "
                f"metadata filters to narrow retrieval scope."
            )
        elif lowest_metric == "context_recall":
            diagnosis = (
                f"Context recall is lowest ({lowest_score:.2f}) — "
                f"relevant chunks are missing from retrieval. "
                f"Consider increasing top_k from 5 to 8, or reducing "
                f"chunk_size to capture more granular matches."
            )
        elif lowest_metric == "faithfulness":
            diagnosis = (
                f"Faithfulness is lowest ({lowest_score:.2f}) — "
                f"the generated answer contains claims not supported by "
                f"the retrieved context. Strengthen the grounding prompt "
                f"to enforce strict adherence to retrieved excerpts only."
            )
        elif lowest_metric == "answer_relevancy":
            diagnosis = (
                f"Answer relevancy is lowest ({lowest_score:.2f}) — "
                f"the generated answer does not directly address the "
                f"question. Review the generation prompt to ensure it "
                f"produces focused, on-topic responses."
            )
        else:
            diagnosis = (
                f"All RAGAS metrics are acceptable. No specific fix needed."
            )

        scores["diagnosis"] = diagnosis

        print(f"\n[ragas_eval] RAGAS Scores:")
        for metric, score in scores.items():
            if metric != "diagnosis":
                print(f"  {metric}: {score:.2f}")
        print(f"  diagnosis: {diagnosis}")

        return scores

    except Exception as e:
        print(f"[ragas_eval] ERROR during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "diagnosis": f"RAGAS evaluation failed: {e}",
        }


def main():
    print("=" * 60)
    print("Step D (RAGAS): Compute RAGAS metrics for dimension 08")
    print("=" * 60)

    scores = compute_ragas_scores()

    # Save RAGAS scores to a separate file for report.py to consume
    ragas_output_path = _repo_root / "ragas_scores.json"
    with open(ragas_output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\n[ragas_eval] ✓ Saved RAGAS scores to {ragas_output_path}")
    return scores


if __name__ == "__main__":
    main()