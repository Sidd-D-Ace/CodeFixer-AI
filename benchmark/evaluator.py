"""
Benchmark Evaluation Harness for CodeGraph-AI.
Directly implements the comparative evaluation methodology detailed in TAE-1:
Compares Single-Shot Baseline (pass@1) vs. LangGraph Self-Repair Orchestrated pipeline.
"""

import time
from typing import Dict, Any, List, Callable, Optional
import pandas as pd
from graph.workflow import build_code_graph
from graph.nodes import coder_node, sandbox_node
from .problems import BENCHMARK_PROBLEMS

def run_single_shot_test(problem: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluates baseline single forward pass (pass@1) without critique or repair loop.
    """
    start_time = time.time()
    
    initial_state = {
        "problem_statement": problem["problem_statement"],
        "unit_tests": problem["unit_tests"],
        "entry_point": problem["entry_point"],
        "max_iterations": 1,
        "generated_code": "",
        "test_results": {},
        "iterations": 0,
        "critique_notes": "",
        "history": [],
        "human_approved": False,
        "human_feedback": None,
        "completed": False
    }

    # Node 1: Code generation
    coder_out = coder_node(initial_state)
    initial_state.update(coder_out)

    # Node 2: Sandbox execution
    sandbox_out = sandbox_node(initial_state)
    initial_state.update(sandbox_out)

    elapsed = time.time() - start_time
    test_res = initial_state.get("test_results", {})

    return {
        "id": problem["id"],
        "title": problem["title"],
        "mode": "Single-Shot Baseline",
        "passed": test_res.get("passed", False),
        "status": test_res.get("status", "FAIL"),
        "iterations": 1,
        "time_sec": round(elapsed, 2),
        "error_message": test_res.get("error_message", "")
    }

def run_langgraph_repair_test(
    problem: Dict[str, str],
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Evaluates LangGraph cyclical self-repair workflow.
    """
    start_time = time.time()
    graph = build_code_graph(enable_interrupt=False)

    initial_state = {
        "problem_statement": problem["problem_statement"],
        "unit_tests": problem["unit_tests"],
        "entry_point": problem["entry_point"],
        "max_iterations": max_iterations,
        "generated_code": "",
        "test_results": {},
        "iterations": 0,
        "critique_notes": "",
        "history": [],
        "human_approved": False,
        "human_feedback": None,
        "completed": False
    }

    config = {"configurable": {"thread_id": f"bench_{problem['id']}_{int(time.time())}"}}
    final_state = graph.invoke(initial_state, config=config)

    elapsed = time.time() - start_time
    test_res = final_state.get("test_results", {})
    history = final_state.get("history", [])

    return {
        "id": problem["id"],
        "title": problem["title"],
        "mode": "LangGraph Orchestrated",
        "passed": test_res.get("passed", False),
        "status": test_res.get("status", "FAIL"),
        "iterations": len(history),
        "time_sec": round(elapsed, 2),
        "error_message": test_res.get("error_message", ""),
        "history": history
    }

def run_full_benchmark_comparison(
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Runs the complete benchmark suite across both modes and compiles comparative statistics.
    """
    total = len(BENCHMARK_PROBLEMS)
    single_shot_results = []
    orchestrated_results = []

    for idx, prob in enumerate(BENCHMARK_PROBLEMS):
        if progress_callback:
            progress_callback(idx + 1, total, f"Testing {prob['title']} ({prob['id']})...")

        # 1. Single-shot run
        ss_res = run_single_shot_test(prob)
        single_shot_results.append(ss_res)

        # 2. LangGraph repair run
        lg_res = run_langgraph_repair_test(prob, max_iterations=3)
        orchestrated_results.append(lg_res)

    # Compute metrics
    ss_passed = sum(1 for r in single_shot_results if r["passed"])
    lg_passed = sum(1 for r in orchestrated_results if r["passed"])

    ss_acc = round((ss_passed / total) * 100, 1)
    lg_acc = round((lg_passed / total) * 100, 1)
    gain = round(lg_acc - ss_acc, 1)

    # Combine into comparison table
    rows = []
    for ss, lg in zip(single_shot_results, orchestrated_results):
        rows.append({
            "Problem ID": ss["id"],
            "Problem Title": ss["title"],
            "Single-Shot Pass": "✅ Pass" if ss["passed"] else "❌ Fail",
            "LangGraph Pass": "✅ Pass" if lg["passed"] else "❌ Fail",
            "Repairs Needed": lg["iterations"],
            "Baseline Time (s)": ss["time_sec"],
            "Orchestrated Time (s)": lg["time_sec"],
            "Resolved via Repair": "Yes" if (not ss["passed"] and lg["passed"]) else ("Direct" if ss["passed"] and lg["passed"] else "No")
        })

    df = pd.DataFrame(rows)

    summary = {
        "total_problems": total,
        "single_shot_accuracy": ss_acc,
        "langgraph_accuracy": lg_acc,
        "accuracy_gain": gain,
        "resolved_via_repair_count": sum(1 for r in rows if r["Resolved via Repair"] == "Yes"),
        "results_dataframe": df,
        "raw_single_shot": single_shot_results,
        "raw_orchestrated": orchestrated_results
    }

    return summary
