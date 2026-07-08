# evaluation/report.py
# Step D — Compiles judged_results.json and the RAGAS scores into
# evaluation_report.json matching the exact structure in spec.md section 9.
# Also provides a __main__ block that runs the full pipeline end-to-end.

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_CASES_PATH = _repo_root / "test_cases.json"
TEST_RESULTS_PATH = _repo_root / "test_results.json"
JUDGED_RESULTS_PATH = _repo_root / "judged_results.json"
RAGAS_SCORES_PATH = _repo_root / "ragas_scores.json"
EVALUATION_REPORT_PATH = _repo_root / "evaluation_report.json"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list | dict:
    """Load a JSON file, returning empty list/dict on failure."""
    if not path.exists():
        print(f"[report] WARNING: {path} not found.")
        return [] if path.suffix != ".json" else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[report] WARNING: Failed to load {path}: {e}")
        return [] if path.suffix != ".json" else {}


# ---------------------------------------------------------------------------
# Dimension name mapping
# ---------------------------------------------------------------------------

DIMENSION_NAMES: dict[str, str] = {
    "01": "functional",
    "02": "quality",
    "03": "safety",
    "04": "security",
    "05": "robustness",
    "06": "performance",
    "07": "context",
    "08": "ragas",
}


def get_dimension_id(dim: str) -> str:
    """Convert dimension name to its ID prefix (01-08)."""
    reverse_map = {v: k for k, v in DIMENSION_NAMES.items()}
    return reverse_map.get(dim, "00")


# ---------------------------------------------------------------------------
# Build the evaluation report
# ---------------------------------------------------------------------------

def compile_report() -> dict:
    """Compile evaluation_report.json matching spec.md section 9."""
    print("=" * 60)
    print("Step D (Report): Compiling evaluation report")
    print("=" * 60)

    # Load all inputs
    judged_results: list[dict] = load_json(JUDGED_RESULTS_PATH)  # type: ignore
    ragas_scores: dict = load_json(RAGAS_SCORES_PATH)  # type: ignore

    if not judged_results:
        print("[report] ERROR: No judged results found. Run judge.py first.")
        # Return a minimal report
        return _build_minimal_report(ragas_scores)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = len(judged_results)
    passed = sum(1 for r in judged_results if r.get("judgement") == "pass")
    failed = sum(1 for r in judged_results if r.get("judgement") == "fail")
    warning = sum(1 for r in judged_results if r.get("judgement") == "warning")

    pass_rate = f"{int((passed / total) * 100)}%" if total > 0 else "0%"

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "warning": warning,
        "pass_rate": pass_rate,
    }

    # -----------------------------------------------------------------------
    # Per-dimension breakdown
    # -----------------------------------------------------------------------
    per_dimension: dict[str, str] = {}

    # Group by dimension
    dim_groups: dict[str, list[dict]] = {}
    for r in judged_results:
        dim = r.get("dimension", "unknown")
        if dim not in dim_groups:
            dim_groups[dim] = []
        dim_groups[dim].append(r)

    for dim_id in sorted(DIMENSION_NAMES.keys()):
        dim_name = DIMENSION_NAMES[dim_id]
        cases = dim_groups.get(dim_name, [])
        if cases:
            dim_passed = sum(1 for r in cases if r.get("judgement") == "pass")
            dim_total = len(cases)
            per_dimension[f"{dim_id}_{dim_name}"] = f"{dim_passed}/{dim_total}"
        else:
            per_dimension[f"{dim_id}_{dim_name}"] = "0/0"

    # Also add any extra dimensions not in the standard 8
    for dim_name in dim_groups:
        dim_id = get_dimension_id(dim_name)
        if f"{dim_id}_{dim_name}" not in per_dimension:
            cases = dim_groups[dim_name]
            dim_passed = sum(1 for r in cases if r.get("judgement") == "pass")
            dim_total = len(cases)
            per_dimension[f"{dim_id}_{dim_name}"] = f"{dim_passed}/{dim_total}"

    # -----------------------------------------------------------------------
    # Weakest dimension
    # -----------------------------------------------------------------------
    weak_dim = _find_weakest_dimension(dim_groups)
    weakest_dimension = weak_dim["name"] if weak_dim else "unknown"

    # -----------------------------------------------------------------------
    # Recommended fix
    # -----------------------------------------------------------------------
    recommended_fix = _generate_recommended_fix(
        weakest_dimension, dim_groups, ragas_scores
    )

    # -----------------------------------------------------------------------
    # RAGAS scores
    # -----------------------------------------------------------------------
    ragas_output = {
        "faithfulness": ragas_scores.get("faithfulness", 0.0),
        "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
        "context_precision": ragas_scores.get("context_precision", 0.0),
        "context_recall": ragas_scores.get("context_recall", 0.0),
    }
    ragas_diagnosis = ragas_scores.get(
        "diagnosis",
        "RAGAS evaluation was not run or did not produce scores.",
    )

    # -----------------------------------------------------------------------
    # Assemble report (spec.md §9 exact structure)
    # -----------------------------------------------------------------------
    report = {
        "summary": summary,
        "per_dimension": per_dimension,
        "weakest_dimension": weakest_dimension,
        "recommended_fix": recommended_fix,
        "ragas_scores": ragas_output,
        "ragas_diagnosis": ragas_diagnosis,
    }

    # Save
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[report] ✓ Saved evaluation report to {EVALUATION_REPORT_PATH}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:      {summary['total']}")
    print(f"  Passed:     {summary['passed']}")
    print(f"  Failed:     {summary['failed']}")
    print(f"  Warnings:   {summary['warning']}")
    print(f"  Pass rate:  {summary['pass_rate']}")
    print(f"  Weakest:    {weakest_dimension}")
    print(f"  Fix:        {recommended_fix}")
    print(f"\n  Per dimension:")
    for dim_key in sorted(per_dimension.keys()):
        print(f"    {dim_key}: {per_dimension[dim_key]}")
    print(f"\n  RAGAS Scores:")
    for metric, score in ragas_output.items():
        print(f"    {metric}: {score:.2f}")
    print(f"  RAGAS Diagnosis: {ragas_diagnosis}")

    return report


def _find_weakest_dimension(dim_groups: dict[str, list[dict]]) -> dict | None:
    """Find the dimension with the lowest pass rate."""
    if not dim_groups:
        return None

    weakest = None
    lowest_rate = 1.0

    for dim_name, cases in dim_groups.items():
        if not cases:
            continue
        dim_passed = sum(1 for r in cases if r.get("judgement") == "pass")
        rate = dim_passed / len(cases)
        if rate < lowest_rate:
            lowest_rate = rate
            weakest = {"name": dim_name, "pass_rate": rate, "passed": dim_passed, "total": len(cases)}

    return weakest


def _generate_recommended_fix(
    weakest_dimension: str,
    dim_groups: dict[str, list[dict]],
    ragas_scores: dict,
) -> str:
    """Generate a specific, actionable fix recommendation."""
    if not weakest_dimension or weakest_dimension == "unknown":
        return "Run the full evaluation pipeline to identify the weakest dimension."

    fix_suggestions = {
        "functional": (
            "Enhance the generation prompt to enforce stricter citation formatting "
            "and ensure complete coverage of multi-part questions by adding an "
            "instruction to address all sub-questions in order."
        ),
        "quality": (
            "Strengthen the grounding instruction to prohibit any information not "
            "present in the retrieved chunks, and add a post-generation fact-check "
            "step that verifies each claim against the source excerpts."
        ),
        "safety": (
            "Add explicit refusal rules for outcome guarantees, comparative rankings, "
            "and medical/legal/financial advice. Include a disclaimer template for "
            "any borderline topics that may arise."
        ),
        "security": (
            "Strengthen system prompt with injection-defence instructions: "
            "instruct the model to ignore any user messages that attempt to override "
            "its role, reveal the system prompt, or assume an alternative persona."
        ),
        "robustness": (
            "Add input validation and sanitization before passing user queries to "
            "the retrieval pipeline. For empty or gibberish input, return the standard "
            "refusal without making an LLM call to save cost and avoid erratic responses."
        ),
        "performance": (
            "Optimize retrieval latency by reducing top_k from 5 to 3 for simple "
            "queries, and consider caching frequent queries with an LRU cache to "
            "avoid redundant embedding calls."
        ),
        "context": (
            "Improve multi-turn context handling by explicitly passing the full "
            "conversation history with clear role markers. Add a prompt instruction "
            "to resolve pronoun references (e.g. 'it', 'the first one') using prior turns."
        ),
        "ragas": (
            f"Address RAGAS metrics: {ragas_scores.get('diagnosis', 'run RAGAS eval for details')}. "
            f"Adjust chunking parameters or retrieval settings based on the lowest metric."
        ),
    }

    # Also check RAGAS scores for additional insights
    if weakest_dimension == "ragas":
        return fix_suggestions.get("ragas", fix_suggestions["ragas"])

    return fix_suggestions.get(weakest_dimension, (
        f"Review and improve the {weakest_dimension} dimension by "
        f"analyzing the specific test cases that failed and updating "
        f"the system prompt or retrieval pipeline accordingly."
    ))


def _build_minimal_report(ragas_scores: dict) -> dict:
    """Build a minimal report when judged results are missing."""
    report = {
        "summary": {"total": 0, "passed": 0, "failed": 0, "warning": 0, "pass_rate": "0%"},
        "per_dimension": {},
        "weakest_dimension": "unknown",
        "recommended_fix": "Run the full evaluation pipeline (generate_tests → run_tests → judge → ragas_eval → report).",
        "ragas_scores": {
            "faithfulness": ragas_scores.get("faithfulness", 0.0),
            "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
            "context_precision": ragas_scores.get("context_precision", 0.0),
            "context_recall": ragas_scores.get("context_recall", 0.0),
        },
        "ragas_diagnosis": ragas_scores.get("diagnosis", "RAGAS evaluation not run."),
    }

    # Still save it
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[report] ✓ Saved minimal evaluation report to {EVALUATION_REPORT_PATH}")
    return report


# ---------------------------------------------------------------------------
# Orchestrator: runs the full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Run the entire evaluation pipeline end-to-end.

    Sequence: generate_tests → run_tests → judge → ragas_eval → report
    """
    print("\n" + "=" * 70)
    print("  BVRIT CHATBOT — FULL EVALUATION PIPELINE")
    print("=" * 70 + "\n")

    # Step A: Generate test cases
    print("[pipeline] Step A: Generating 20 test cases …")
    from evaluation.generate_tests import main as generate_main
    test_cases = generate_main()
    print("[pipeline] ✓ Step A complete\n")

    # Step B: Run tests
    print("[pipeline] Step B: Running test cases against chatbot …")
    from evaluation.run_tests import main as run_tests_main
    test_results = run_tests_main()
    print("[pipeline] ✓ Step B complete\n")

    # Step C: Judge results
    print("[pipeline] Step C: LLM-as-judge evaluation …")
    from evaluation.judge import main as judge_main
    judged_results = judge_main()
    print("[pipeline] ✓ Step C complete\n")

    # Step D (RAGAS): Compute RAGAS metrics
    print("[pipeline] Step D (RAGAS): Computing RAGAS metrics …")
    from evaluation.ragas_eval import main as ragas_main
    ragas_scores = ragas_main()
    print("[pipeline] ✓ Step D (RAGAS) complete\n")

    # Step D (Report): Compile final report
    print("[pipeline] Step D (Report): Compiling evaluation report …")
    report = compile_report()
    print("[pipeline] ✓ Step D (Report) complete\n")

    print("=" * 70)
    print("  PIPELINE COMPLETE — evaluation_report.json generated")
    print("=" * 70)

    return report


# ---------------------------------------------------------------------------
# __main__ — entry point for both standalone and orchestrated runs
# ---------------------------------------------------------------------------

def main():
    """Run just the report compilation step (assumes prior steps done)."""
    compile_report()


if __name__ == "__main__":
    # When run directly via `python evaluation/report.py`, run the full pipeline
    run_full_pipeline()