"""
Sandbox execution module for CodeGraph-AI.
Safely runs generated code against unit test assertions in an isolated subprocess with timeout protection.
"""

import sys
import os
import tempfile
import subprocess
from typing import Dict, Any, Optional

def execute_code_with_tests(
    code: str,
    unit_tests: str,
    timeout_seconds: int = 8
) -> Dict[str, Any]:
    """
    Executes Python code combined with unit test assertions in a dedicated subprocess.
    
    Args:
        code: The Python function or implementation code.
        unit_tests: Test cases and assert statements.
        timeout_seconds: Max execution duration before terminating.
        
    Returns:
        Dict containing:
            - status: "PASS", "FAIL", "TIMEOUT", or "ERROR"
            - passed: bool
            - stdout: standard output
            - stderr: traceback / error details
            - error_message: concise summary of the failure
    """
    # Combine user code and test suite
    full_script = (
        "# --- Generated Code ---\n"
        f"{code.strip()}\n\n"
        "# --- Unit Tests ---\n"
        f"{unit_tests.strip()}\n"
    )

    temp_file = None
    try:
        # Create temporary execution script
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            temp_file = f.name
            f.write(full_script)

        # Run in separate python process
        cmd = [sys.executable, temp_file]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace"
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode == 0:
            return {
                "status": "PASS",
                "passed": True,
                "stdout": stdout,
                "stderr": "",
                "error_message": ""
            }
        else:
            # Distinguish AssertionError vs SyntaxError/RuntimeError
            is_assertion = "AssertionError" in stderr
            status = "FAIL" if is_assertion else "ERROR"
            
            # Extract last line of traceback for concise summary
            lines = [l for l in stderr.splitlines() if l.strip()]
            summary = lines[-1] if lines else "Non-zero exit code"

            return {
                "status": status,
                "passed": False,
                "stdout": stdout,
                "stderr": stderr,
                "error_message": summary
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "passed": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds (potential infinite loop).",
            "error_message": f"TimeoutExpired ({timeout_seconds}s limit)"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "passed": False,
            "stdout": "",
            "stderr": str(e),
            "error_message": f"Execution exception: {str(e)}"
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
