"""Hidden reference tests = the objective ground truth for the fizzbuzz task.
The agent never sees these; its solution.py is run against them in a sandbox.
All expected values are STRINGS (the task asks for str(n) for plain numbers),
so an int-returning solution fails several cases."""
import pytest

from solution import fizzbuzz

CASES = [
    (1, "1"), (2, "2"), (3, "Fizz"), (4, "4"), (5, "Buzz"), (6, "Fizz"),
    (7, "7"), (9, "Fizz"), (10, "Buzz"), (15, "FizzBuzz"), (30, "FizzBuzz"), (45, "FizzBuzz"),
]


@pytest.mark.parametrize("n,expected", CASES)
def test_fizzbuzz(n, expected):
    assert fizzbuzz(n) == expected
