"""One-judge smoke for the FreeStack jury (verdict + LM Studio, zero cost).

Goal: prove verdict can drive a LOCAL judge model through LM Studio and return a
structured score. The open question this answers empirically: verdict patches
litellm with instructor in Mode.TOOLS for the `lm_studio` provider (see
verdict/model.py ClientWrapper) - local GGUF models may not do tool-calls, same
class of gotcha the agent hit. If this run returns a score, TOOLS mode works and
we can build the N-judge panel; if it raises, we force JSON mode.

Env (source ../env-wsl.sh first):
  LM_STUDIO_API_BASE=http://<host>:1234/v1
  LM_STUDIO_API_KEY=<placeholder ok, auth is open>

Usage:
  JUDGE_MODEL=lm_studio/phi-4-14b python jury_smoke.py
"""

import os
import sys
import traceback

# --- GOTCHA FIX: force instructor JSON mode for the local LM Studio endpoint ---
# verdict patches litellm with instructor in Mode.TOOLS for the `lm_studio` provider
# (verdict/model.py ClientWrapper, hardcoded local dict). In TOOLS mode instructor
# sends tool_choice as an OBJECT, but LM Studio only accepts tool_choice as a string
# -> HTTP 400 "Invalid tool_choice type: 'object'". JSON mode uses response_format
# instead, which LM Studio supports. We wrap instructor.patch to override the mode.
# verdict does `from instructor import Mode, patch` lazily inside __init__, so patching
# the module attribute before the pipeline runs is enough.
import instructor

_INSTRUCTOR_MODE = os.getenv("INSTRUCTOR_MODE", "JSON")  # JSON | MD_JSON | JSON_SCHEMA | TOOLS
if _INSTRUCTOR_MODE != "TOOLS":
    _forced_mode = getattr(instructor.Mode, _INSTRUCTOR_MODE)
    _orig_patch = instructor.patch

    def _patched_patch(*a, **k):
        k["mode"] = _forced_mode
        return _orig_patch(*a, **k)

    instructor.patch = _patched_patch

from verdict import Pipeline
from verdict.common.judge import JudgeUnit
from verdict.scale import ContinuousScale
from verdict.schema import Schema
from verdict.util import ratelimit

ratelimit.disable()  # local single-endpoint, no provider rate limit to respect

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "lm_studio/phi-4-14b")

JUDGE_PROMPT = """You are a strict code reviewer grading a candidate answer.

Question:
{source.question}

Candidate answer:
{source.answer}

Score the answer from 1 (wrong/unusable) to 5 (correct, clean, complete).
Give a one-sentence explanation, then the score."""

# A clearly gradeable sample: a correct, clean answer should score high (4-5).
SAMPLE = Schema.of(
    question="Write a Python function is_even(n) that returns True if n is even.",
    answer="def is_even(n):\n    return n % 2 == 0",
)

print(f"judge model : {JUDGE_MODEL}")
print(f"api base    : {os.getenv('LM_STUDIO_API_BASE')}")
print("running one-judge pipeline (first call JIT-loads the model in LM Studio)...\n")

pipeline = (
    Pipeline("jurysmoke")
    >> JudgeUnit(scale=ContinuousScale(1, 5), explanation=True)
    .prompt(JUDGE_PROMPT)
    .via(JUDGE_MODEL, retries=2, timeout=600)
)

try:
    df, cols = pipeline.run_from_list([SAMPLE])
    print("OK - leaf columns:", cols)
    for c in cols:
        print(f"  {c} = {df[c].iloc[0]!r}")
    # surface explanation column if present
    expl = [c for c in df.columns if c.endswith("_explanation")]
    for c in expl:
        print(f"  {c} = {str(df[c].iloc[0])[:300]!r}")
    print("\nRESULT: JUDGE_OK")
except Exception as e:  # noqa: BLE001 - smoke, surface whatever happens
    print("RESULT: JUDGE_FAILED")
    print("  exc type:", type(e).__name__)
    print("  exc msg :", str(e)[:500])
    print("\n--- traceback tail ---")
    traceback.print_exc()
    sys.exit(1)
