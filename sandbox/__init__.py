"""
Sandbox package for safe execution of LLM-generated code.
"""
from .runner import execute_code_with_tests

__all__ = ["execute_code_with_tests"]
