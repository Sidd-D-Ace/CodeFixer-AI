"""
Optimized LangGraph workflow builder for CodeFixer AI.
Constructs a high-performance cyclical StateGraph containing unified synthesis,
sandboxed execution, and self-healing repair loops.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    synthesizer_node,
    sandbox_node,
    debug_repair_node,
    finalize_node
)

def route_after_sandbox(state: AgentState) -> Literal["debug_repair", "finalize"]:
    """
    Conditional routing function (LangGraph edge logic):
    - If all unit test assertions pass -> route to 'finalize'.
    - If tests fail AND iterations < max_iterations -> route to 'debug_repair' (cycle back).
    - If tests fail AND max_iterations reached -> route to 'finalize' (stop with best effort).
    """
    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)

    if passed:
        return "finalize"

    if iteration < max_iterations:
        return "debug_repair"
    else:
        return "finalize"

def build_codefixer_graph():
    """
    Compiles and returns the LangGraph stateful workflow with checkpointing.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Core Nodes
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("sandbox", sandbox_node)
    workflow.add_node("debug_repair", debug_repair_node)
    workflow.add_node("finalize", finalize_node)

    # 2. Add Linear Ingestion Edge
    workflow.add_edge(START, "synthesizer")
    workflow.add_edge("synthesizer", "sandbox")

    # 3. Add Cyclical Conditional Edge from Sandbox
    workflow.add_conditional_edges(
        "sandbox",
        route_after_sandbox,
        {
            "debug_repair": "debug_repair",
            "finalize": "finalize"
        }
    )

    # 4. Feedback Loop: Debugger routes back into Sandbox for re-testing
    workflow.add_edge("debug_repair", "sandbox")
    workflow.add_edge("finalize", END)

    # 5. Checkpointer for step-by-step state inspection
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
