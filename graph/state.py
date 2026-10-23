"""
State schema for CodeFixer AI agent.
Defines the shared state dictionary passed across nodes in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional

class IterationRecord(TypedDict):
    iteration: int
    code: str
    status: str
    passed: bool
    error_message: str
    stderr: str
    critique: str

class AgentState(TypedDict):
    """
    Central state object threaded through the LangGraph StateGraph.
    Tracks user input, generated/repaired code, test cases, and repair loop iterations.
    """
    # Inputs
    mode: str                         # "prompt" (create from scratch) or "debug" (fix provided code)
    user_input: str                   # Natural language specification OR buggy source code
    custom_tests: Optional[str]       # Optional user-supplied test cases
    max_iterations: int               # Maximum self-repair retry budget (e.g. 3-5)
    model_name: str                   # LLM model name

    # Internal Pipeline State
    problem_spec: str                 # Extracted problem definition
    unit_tests: str                   # Complete suite of assert statements (auto-generated or custom)
    tests_generated: bool             # Whether test suite was auto-synthesized by agent
    current_code: str                 # Latest synthesized / repaired Python code
    thought_process: str              # Step-by-step thinking (intent, edge cases, responsible AI safety)
    explanation: str                  # Articulate, modern natural language response
    thinking_time_sec: float          # Time elapsed during model reasoning
    test_results: Dict[str, Any]      # Subprocess execution output (status, passed, stdout, stderr, etc.)
    iteration: int                    # Current repair attempt counter
    critique_notes: str               # Debugger diagnostic reasoning
    history: List[IterationRecord]    # Snapshot of every code version & test outcome
    logs: List[str]                   # Human-readable progress logs with timestamps
    active_node: str                  # Current executing LangGraph node
    success: bool                     # True if code passes all assertions
