# CodeFixer AI
### Autonomous Code Synthesis & Self-Healing Debugging Agent Powered by LangGraph & Local Ollama

CodeFixer AI is an autonomous software engineering assistant that generates verified, bug-free Python code from natural language prompts, or takes existing buggy code and repairs it. It runs candidate code through automated unit tests in an isolated execution sandbox and uses a **cyclical LangGraph state machine** to inspect error tracebacks, diagnose root causes, and self-repair until all assertions pass.

---

## 🧭 Where and How is LangGraph Used?

Unlike linear LLM chains that execute a single prompt and produce a one-off response (pass@1), CodeFixer AI uses **LangGraph** to build a **stateful, cyclical, multi-node agent**:

```
                       ┌─────────────────────────┐
                       │  analyze_input_node     │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  test_synthesizer_node  │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  code_generator_node    │
                       └────────────┬────────────┘
                                    │
                                    ▼
       ┌────────────────►┌───────────────────────┐
       │                 │     sandbox_node      │ (Executes in isolated subprocess)
       │                 └──────────┬────────────┘
       │                            │
(Loop if tests fail)         [Conditional Edge]
       │                    /                  \
       │         (Tests Failed)              (All Passed)
       │                  │                      │
┌──────┴──────────────┐   │                      ▼
│  debug_repair_node  │◄──┘             ┌─────────────────┐
│ (Critique & Patch)  │                 │  finalize_node  │ ──► END (Verified Code)
└─────────────────────┘                 └─────────────────┘
```

### LangGraph Architectural Components:
1. **`StateGraph` & Shared State Schema (`AgentState`)**:
   - Manages state threading across nodes: problem specifications, candidate code, unit test assertions, subprocess return codes, and error tracebacks.
2. **Cyclic Conditional Edges (`add_conditional_edges`)**:
   - The edge from `sandbox_node` routes conditionally:
     - If `test_results['passed'] == True` $\to$ proceeds to `finalize_node`.
     - If assertions fail or syntax/runtime errors occur and attempts remain $\to$ routes to `debug_repair_node`.
3. **Self-Healing Feedback Loop**:
   - `debug_repair_node` analyzes the exact traceback (e.g. `IndexError`, `AssertionError`), updates the state with diagnostic critique notes, and routes **back to `sandbox_node`** for automated re-testing.
4. **State Persistence (`MemorySaver` checkpointer)**:
   - Snapshots every iteration's code, error message, and diagnosis, enabling inspection of how the code evolved from buggy to verified.

---

## 🚀 Features

- **Prompt-Based Generation**: Enter any function description; the agent synthesizes the code, writes comprehensive edge-case test cases, and verifies them.
- **Buggy Code Repair**: Paste buggy code; the agent diagnoses syntax/logic errors, tests boundary cases, and repairs the code.
- **Isolated Sandbox Execution**: Runs tests in a subprocess with strict timeout guards (preventing infinite loops).
- **100% Local Execution**: Runs on your local GPU via **Ollama** (`qwen2.5-coder:1.5b`) with zero cloud API costs.
- **Standard Web UI**: Modern, responsive dark-mode frontend built with standard HTML5, CSS3, and JavaScript served via FastAPI.

---

## ⚡ Quick Start

### 1. Ensure Ollama is Running
```powershell
ollama run qwen2.5-coder:1.5b
```

### 2. Activate Environment & Run Server
```powershell
.\venv\Scripts\activate
python server.py
```

### 3. Open in Browser
Visit:
👉 **`http://127.0.0.1:8000`**
