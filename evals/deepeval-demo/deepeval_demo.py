"""DeepEval smoke - a G-Eval metric judged by a LOCAL LM Studio model, zero cost.

DeepEval ships LiteLLMModel, which sets response_format=<pydantic schema> and lets
litellm convert it to a json_schema request - exactly what LM Studio accepts (the
same path our jury uses; NOT the instructor-TOOLS path that broke verdict).

G-Eval = the judge writes evaluation steps from your criteria (chain-of-thought),
then scores the output 0..1 with a reason. We run it on a correct answer and a
buggy one to show the metric discriminates.

Run: cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
       && cd evals/deepeval-demo \
       && DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python deepeval_demo.py
"""

import os

from deepeval.metrics import GEval
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

JUDGE = os.getenv("DEEPEVAL_JUDGE", "lm_studio/phi-4-14b")  # different family from any agent

judge = LiteLLMModel(
    model=JUDGE,
    base_url=os.getenv("LM_STUDIO_API_BASE"),  # http://<host>:1234/v1
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    temperature=0,
)
print(f"judge model: {JUDGE}  via {os.getenv('LM_STUDIO_API_BASE')}\n")

correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output is a correct and complete answer to the "
        "question in the input. Penalize logic errors heavily."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.5,
)

QUESTION = "Write a Python function is_even(n) that returns True if n is even."
CASES = [
    ("good", "def is_even(n):\n    return n % 2 == 0"),
    ("buggy", "def is_even(n):\n    return n % 2 == 1  # True for ODD numbers"),
]

print("=== G-Eval: Correctness (judged locally) ===")
results = []
for label, answer in CASES:
    tc = LLMTestCase(input=QUESTION, actual_output=answer)
    correctness.measure(tc)
    results.append((label, correctness.score, correctness.success))
    print(f"  {label:6s} score={correctness.score:.2f} pass={correctness.success}")
    print(f"         reason: {str(correctness.reason)[:160]}")

good = next(s for l, s, _ in results if l == "good")
bad = next(s for l, s, _ in results if l == "buggy")
print(f"\nDISCRIMINATION: good({good:.2f}) {'>' if good > bad else '<='} buggy({bad:.2f}) "
      f"-> {'PASS' if good > bad else 'FAIL'}")
