"""FreeStack jury v1 - a panel of LOCAL judges scoring candidate answers, zero cost.

Why direct litellm instead of the `verdict` framework:
  verdict 0.2.7 hard-pins instructor==1.7.2, whose retry path breaks against the
  installed tenacity 9.x ("Retrying.__call__() got multiple values for 'self'"),
  and whose JSON modes emit response_format=json_object, which LM Studio rejects
  ("'response_format.type' must be 'json_schema' or 'text'"). Direct litellm with
  response_format=json_schema is what LM Studio actually supports (grammar-constrained
  structured output) and is fully under our control. See LOCAL-RUN.md gotchas #4/#5.

Design (per project rules + Ensemble-Aggregation research):
  - Judges are DIFFERENT families (agent qwen-coder is NEVER a judge).
  - Run SEQUENTIALLY: 12 GB VRAM holds one <=14B model at a time (LM Studio JIT-loads).
    Judge is the OUTER loop so each model loads once and scores every case.
  - Each judge returns a schema-validated {explanation, score 1-5} at temperature 0.
  - Aggregate per case: mean / min / max across judges + per-judge breakdown.
  - Verdict JSON is written to jury/runs/ as the session artifact.

Env (source ../env-wsl.sh first): LM_STUDIO_API_BASE, LM_STUDIO_API_KEY.
Override judges: JURY_JUDGES="lm_studio/phi-4-14b,lm_studio/..." python jury.py
"""

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

litellm.drop_params = False
litellm.suppress_debug_info = True

# Judges: 5 decorrelated families, all official instruct (agent qwen-coder is NEVER here).
# Note: ministral/deepseek are reasoning models - under json_schema grammar they cannot
# emit <think>, so CoT is suppressed (fine for scoring, watch calibration).
DEFAULT_JUDGES = [
    "lm_studio/phi-4-14b",                             # Microsoft
    "lm_studio/mistralai/ministral-3-14b-reasoning",   # Mistral
    "lm_studio/llama-3.3-8b-instruct-i1",              # Meta
    "lm_studio/gemma-4-e4b-it",                         # Google
    "lm_studio/deepseek-r1-distill-qwen-14b",          # DeepSeek
]
JUDGES = [m.strip() for m in os.getenv("JURY_JUDGES", ",".join(DEFAULT_JUDGES)).split(",") if m.strip()]

SCHEMA = {
    "name": "review_score",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["explanation", "score"],
        "additionalProperties": False,
    },
}

RUBRIC = (
    "You are a strict code reviewer. Grade the candidate answer to the question on a "
    "1-5 scale: 1 = wrong/unusable, 3 = works but flawed, 5 = correct, clean, complete. "
    "Judge correctness first. Reply ONLY with the JSON object."
)

# Two cases that a real jury must tell apart: a correct answer and a subtly WRONG one.
CASES = [
    {
        "id": "is_even_correct",
        "question": "Write a Python function is_even(n) that returns True if n is even.",
        "answer": "def is_even(n):\n    return n % 2 == 0",
        "expect": "high",
    },
    {
        "id": "is_even_buggy",
        "question": "Write a Python function is_even(n) that returns True if n is even.",
        "answer": "def is_even(n):\n    return n % 2 == 1  # returns True for ODD numbers",
        "expect": "low",
    },
]


def score_one(model: str, question: str, answer: str) -> dict:
    """One judge scores one (question, answer). Returns {score, explanation} or {error}."""
    msgs = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"Question:\n{question}\n\nCandidate answer:\n{answer}"},
    ]
    t0 = time.time()
    try:
        r = litellm.completion(
            model=model, messages=msgs,
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            temperature=0, timeout=600,
        )
        parsed = json.loads(r.choices[0].message.content)
        return {"score": int(parsed["score"]), "explanation": parsed.get("explanation", ""),
                "secs": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "secs": round(time.time() - t0, 1)}


def aggregate(scores: list) -> dict:
    """Robust aggregation across judges. median resists a single harsh/lenient outlier
    that mean does not; spread/disagree surface jury calibration problems."""
    if not scores:
        return {"n_ok": 0}
    return {
        "n_ok": len(scores),
        "mean": round(statistics.mean(scores), 2),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
        "disagree": (max(scores) - min(scores)) >= 2,  # >=2 points apart = judges disagree
    }


def main() -> None:
    print(f"FreeStack jury v1 | api={os.getenv('LM_STUDIO_API_BASE')}")
    print(f"judges ({len(JUDGES)}, sequential): " + ", ".join(j.split('/')[-1] for j in JUDGES))
    print(f"cases ({len(CASES)}): " + ", ".join(c["id"] for c in CASES) + "\n")

    # judge-outer loop so each model JIT-loads once and scores every case
    results = {c["id"]: {} for c in CASES}
    for model in JUDGES:
        short = model.split("/")[-1]
        print(f"--- judge: {short} (loading + scoring {len(CASES)} cases) ---")
        for case in CASES:
            out = score_one(model, case["question"], case["answer"])
            results[case["id"]][model] = out
            if "error" in out:
                print(f"    {case['id']:18s} ERROR {out['error']}  ({out['secs']}s)")
            else:
                print(f"    {case['id']:18s} score={out['score']}  ({out['secs']}s)  {out['explanation'][:80]}")
        print()

    # aggregate per case (mean + robust median + disagreement flag)
    verdict = {"generated_at": datetime.now(timezone.utc).isoformat(), "judges": JUDGES, "cases": []}
    print("=== VERDICT (mean + median across judges) ===")
    for case in CASES:
        scores = [r["score"] for r in results[case["id"]].values() if "score" in r]
        agg = {
            "id": case["id"], "expect": case["expect"],
            "scores": {m.split('/')[-1]: results[case["id"]][m].get("score", results[case["id"]][m].get("error"))
                       for m in JUDGES},
            **aggregate(scores),
        }
        verdict["cases"].append(agg)
        flag = "  <-- JUDGES DISAGREE" if agg.get("disagree") else ""
        print(f"  {case['id']:18s} expect={case['expect']:4s} "
              f"mean={agg.get('mean','n/a')} median={agg.get('median','-')} "
              f"spread={agg.get('spread','-')}{flag}\n"
              f"      per-judge={agg['scores']}")

    # discrimination check: did the jury rank the correct answer above the buggy one?
    by_id = {c["id"]: c for c in verdict["cases"]}
    if "mean" in by_id.get("is_even_correct", {}) and "mean" in by_id.get("is_even_buggy", {}):
        good, bad = by_id["is_even_correct"]["mean"], by_id["is_even_buggy"]["mean"]
        verdict["discriminates"] = good > bad
        print(f"\nDISCRIMINATION: correct({good}) {'>' if good > bad else '<='} buggy({bad}) "
              f"-> {'PASS' if good > bad else 'FAIL'}")

    runs = Path(__file__).parent / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = runs / f"verdict_{stamp}.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(f"\nverdict saved -> {out_path}")


if __name__ == "__main__":
    main()
