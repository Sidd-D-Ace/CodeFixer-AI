"""
Curated HumanEval-style coding problems for CodeGraph-AI benchmark evaluation.
Each problem contains specification, entry point, and test cases covering typical algorithmic edge cases.
"""

from typing import List, Dict

BENCHMARK_PROBLEMS: List[Dict[str, str]] = [
    {
        "id": "HE-01",
        "title": "Has Close Elements",
        "entry_point": "has_close_elements",
        "problem_statement": """def has_close_elements(numbers: list[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)
    True
    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)
    False
    \"\"\"""",
        "unit_tests": """assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 1.0], 0.0) == False
assert has_close_elements([], 0.5) == False
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-02",
        "title": "Separate Paren Groups",
        "entry_point": "separate_paren_groups",
        "problem_statement": """def separate_paren_groups(paren_string: str) -> list[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"""",
        "unit_tests": """assert separate_paren_groups('(()()) (( )) () ((())()())') == ['(()())', '(())', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(())((())))') == ['(()(())((())))']
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-03",
        "title": "Truncate Number",
        "entry_point": "truncate_number",
        "problem_statement": """def truncate_number(number: float) -> float:
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1, also called fractional part).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"""",
        "unit_tests": """assert abs(truncate_number(3.5) - 0.5) < 1e-6
assert abs(truncate_number(1.25) - 0.25) < 1e-6
assert abs(truncate_number(123.0) - 0.0) < 1e-6
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-04",
        "title": "Below Zero Balance Check",
        "entry_point": "below_zero",
        "problem_statement": """def below_zero(operations: list[int]) -> bool:
    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    \"\"\"""",
        "unit_tests": """assert below_zero([]) == False
assert below_zero([1, 2, -3, 1, 2, -3]) == False
assert below_zero([1, 2, -4, 5, 6]) == True
assert below_zero([1, -1, 2, -2, 5, -5, -1]) == True
assert below_zero([-1, 2, 3]) == True
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-05",
        "title": "Mean Absolute Deviation",
        "entry_point": "mean_absolute_deviation",
        "problem_statement": """def mean_absolute_deviation(numbers: list[float]) -> float:
    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between the numbers
    and a centerpoint (mean):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    \"\"\"""",
        "unit_tests": """assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
assert abs(mean_absolute_deviation([2.0, 2.0, 2.0, 2.0]) - 0.0) < 1e-6
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-06",
        "title": "Intersperse Element",
        "entry_point": "intersperse",
        "problem_statement": """def intersperse(numbers: list[int], delimeter: int) -> list[int]:
    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    \"\"\"""",
        "unit_tests": """assert intersperse([], 7) == []
assert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
assert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]
assert intersperse([1], 100) == [1]
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-07",
        "title": "Filter By Substring",
        "entry_point": "filter_by_substring",
        "problem_statement": """def filter_by_substring(strings: list[str], substring: str) -> list[str]:
    \"\"\" Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bac', 'cba', 'array'], 'a')
    ['abc', 'bac', 'cba', 'array']
    >>> filter_by_substring(['xyz', 'john', 'doe'], 'a')
    []
    \"\"\"""",
        "unit_tests": """assert filter_by_substring([], 'john') == []
assert filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxy', 'xxxAAA', 'xxx']
assert filter_by_substring(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') == ['grunt', 'prune']
print("ALL TESTS PASSED")"""
    },
    {
        "id": "HE-08",
        "title": "Rolling Max Array",
        "entry_point": "rolling_max",
        "problem_statement": """def rolling_max(numbers: list[int]) -> list[int]:
    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    \"\"\"""",
        "unit_tests": """assert rolling_max([]) == []
assert rolling_max([1, 2, 3, 4]) == [1, 2, 3, 4]
assert rolling_max([4, 3, 2, 1]) == [4, 4, 4, 4]
assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]
print("ALL TESTS PASSED")"""
    }
]
