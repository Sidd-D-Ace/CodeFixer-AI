"""
FastAPI Server for CodeFixer AI.
Serves standard HTML/CSS/JS frontend and exposes REST endpoints for LangGraph agent execution.
"""

import os
import uuid
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests

from graph.workflow import build_codefixer_graph
from graph.state import AgentState

app = FastAPI(
    title="CodeFixer AI",
    description="Autonomous Code Generation & Self-Healing Debugger Agent Powered by LangGraph",
    version="1.0.0"
)

# Static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class AgentRequest(BaseModel):
    mode: str = "prompt"  # "prompt" or "debug"
    user_input: str
    custom_tests: Optional[str] = None
    max_iterations: int = 3
    model_name: str = "qwen2.5-coder:1.5b"

@app.get("/")
async def serve_index():
    """Serves the standard HTML/CSS/JS frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CodeFixer AI Backend Running. static/index.html not found."}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/models")
async def list_models():
    """Fetches list of available models from local Ollama instance."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            return {"models": [m["name"] for m in models_data]}
    except Exception:
        pass
    return {"models": ["qwen2.5-coder:1.5b"]}

@app.post("/api/run-agent")
async def run_agent(req: AgentRequest):
    """
    Executes the LangGraph cyclical state machine:
    1. analyze_input_node
    2. test_synthesizer_node
    3. code_generator_node
    4. sandbox_node
    5. (conditional loop) debug_repair_node -> sandbox_node
    6. finalize_node
    """
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    try:
        # Build compiled LangGraph workflow
        graph = build_codefixer_graph()

        initial_state: AgentState = {
            "mode": req.mode,
            "user_input": req.user_input,
            "custom_tests": req.custom_tests,
            "max_iterations": max(1, min(req.max_iterations, 5)),
            "model_name": req.model_name,
            "problem_spec": "",
            "unit_tests": "",
            "tests_generated": False,
            "current_code": "",
            "test_results": {},
            "iteration": 0,
            "critique_notes": "",
            "history": [],
            "logs": [],
            "active_node": "START",
            "success": False
        }

        thread_id = f"exec_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        # Execute the LangGraph StateGraph
        final_state = graph.invoke(initial_state, config=config)

        return {
            "success": final_state.get("success", False),
            "mode": final_state.get("mode", req.mode),
            "problem_spec": final_state.get("problem_spec", ""),
            "current_code": final_state.get("current_code", ""),
            "thought_process": final_state.get("thought_process", ""),
            "explanation": final_state.get("explanation", ""),
            "thinking_time_sec": final_state.get("thinking_time_sec", 0.0),
            "critique_notes": final_state.get("critique_notes", ""),
            "unit_tests": final_state.get("unit_tests", ""),
            "tests_generated": final_state.get("tests_generated", False),
            "test_results": final_state.get("test_results", {}),
            "history": final_state.get("history", []),
            "logs": final_state.get("logs", []),
            "iterations": final_state.get("iteration", 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
