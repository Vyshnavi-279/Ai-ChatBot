# evaluation/run_tests.py
# Step B — Loads test_cases.json, runs each question through the live chatbot
# pipeline (rag_core/retriever.py + rag_core/generation.py), and records:
# question, expected_answer, actual_response, retrieved_chunks, latency_seconds.
# For dimension 07 (context) tests, runs turns sequentially feeding prior turns as history.
# Saves results to test_results.json.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rag_core.retriever import retrieve
from rag_core.generation import generate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_CASES_PATH = _repo_root / "test_cases.json"
TEST_RESULTS_PATH = _repo_root / "test_results.json"


def load_test_cases() -> list[dict]:
    """Load test cases from test_cases.json."""
    if not TEST_CASES_PATH.exists():
        print(f"[run_tests] ERROR: {TEST_CASES_PATH} not found. Run generate_tests.py first.")
        sys.exit(1)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"[run_tests] Loaded {len(test_cases)} test cases from {TEST_CASES_PATH}")
    return test_cases


def run_single_test(test_case: dict) -> dict:
    """Run a single test case through the RAG pipeline and return results.

    For context-dimension tests (dimension == "context"), runs turns sequentially,
    feeding prior turns as history.
    """
    dimension = test_case.get("dimension", "")
    question = test_case.get("question", "")
    expected_answer = test_case.get("expected_answer", "")
    pass_fail_criteria = test_case.get("pass_fail_criteria", {})
    sla_seconds = test_case.get("sla_seconds", 10)

    result = {
        "id": test_case.get("id", "00-000"),
        "dimension": dimension,
        "question": question,
        "expected_answer": expected_answer,
        "pass_fail_criteria": pass_fail_criteria,
        "actual_response": "",
        "retrieved_chunks": [],
        "latency_seconds": 0.0,
        "error": None,
    }

    # Handle context dimension (multi-turn)
    turns = test_case.get("turns", None)
    if turns and isinstance(turns, list) and len(turns) > 1:
        return run_multi_turn_test(test_case, turns)

    # Single-turn test
    try:
        start_time = time.time()

        # Retrieve chunks
        chunks = retrieve(question, top_k=5)

        # Generate answer
        gen_result = generate(query=question, chunks=chunks)

        elapsed = time.time() - start_time

        result["actual_response"] = gen_result.get("answer", "")
        result["retrieved_chunks"] = [
            {
                "text": c["text"][:500],  # truncate for storage
                "metadata": c["metadata"],
                "score": c["score"],
            }
            for c in chunks
        ]
        result["latency_seconds"] = round(elapsed, 3)

    except Exception as e:
        result["error"] = str(e)
        result["actual_response"] = f"ERROR: {e}"
        result["latency_seconds"] = -1

    return result


def run_multi_turn_test(test_case: dict, turns: list[str]) -> dict:
    """Run a multi-turn context test, feeding history between turns."""
    dimension = test_case.get("dimension", "")
    expected_answer = test_case.get("expected_answer", "")
    pass_fail_criteria = test_case.get("pass_fail_criteria", {})

    result = {
        "id": test_case.get("id", "00-000"),
        "dimension": dimension,
        "question": turns[0] if turns else "",
        "expected_answer": expected_answer,
        "pass_fail_criteria": pass_fail_criteria,
        "actual_response": "",
        "retrieved_chunks": [],
        "latency_seconds": 0.0,
        "turns": turns,
        "turn_responses": [],
        "error": None,
    }

    history: list[dict] = []
    total_start = time.time()

    try:
        for i, turn_question in enumerate(turns):
            # Retrieve chunks for this turn
            chunks = retrieve(turn_question, top_k=5)

            # Generate answer with history
            gen_result = generate(
                query=turn_question,
                chunks=chunks,
                history=history,
            )

            answer = gen_result.get("answer", "")

            # Store turn response
            result["turn_responses"].append({
                "turn": i + 1,
                "question": turn_question,
                "response": answer,
                "retrieved_chunks": [
                    {
                        "text": c["text"][:500],
                        "metadata": c["metadata"],
                        "score": c["score"],
                    }
                    for c in chunks
                ],
            })

            # Update history for next turn
            history.append({"role": "user", "content": turn_question})
            history.append({"role": "assistant", "content": answer})

        total_elapsed = time.time() - total_start

        # The "actual_response" is the final turn's answer
        if result["turn_responses"]:
            result["actual_response"] = result["turn_responses"][-1]["response"]
            result["retrieved_chunks"] = result["turn_responses"][-1]["retrieved_chunks"]
        result["latency_seconds"] = round(total_elapsed, 3)

    except Exception as e:
        result["error"] = str(e)
        result["actual_response"] = f"ERROR: {e}"
        result["latency_seconds"] = -1

    return result


def main():
    print("=" * 60)
    print("Step B: Run test cases against chatbot pipeline")
    print("=" * 60)

    test_cases = load_test_cases()
    results: list[dict] = []

    for i, tc in enumerate(test_cases, start=1):
        tc_id = tc.get("id", f"case-{i}")
        dimension = tc.get("dimension", "unknown")
        question_preview = tc.get("question", "")[:60].replace("\n", " ")

        print(f"\n[{i}/{len(test_cases)}] Running {tc_id} ({dimension}): {question_preview}…")

        result = run_single_test(tc)
        results.append(result)

        # Print brief status
        latency = result.get("latency_seconds", -1)
        error = result.get("error")
        if error:
            print(f"  → ERROR: {error}")
        else:
            resp_preview = result.get("actual_response", "")[:80].replace("\n", " ")
            print(f"  → Latency: {latency}s | Response: {resp_preview}…")

    # Save results
    with open(TEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[run_tests] ✓ Saved {len(results)} results to {TEST_RESULTS_PATH}")

    # Summary
    errors = sum(1 for r in results if r.get("error"))
    print(f"[run_tests] Summary: {len(results)} total, {errors} errors")
    return results


if __name__ == "__main__":
    main()