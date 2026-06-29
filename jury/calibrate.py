"""Calibrate the FreeStack jury: measure each judge's bias against a GOLD set.

Today's loop run showed judges disagree by family (phi-4/gemma strict, ministral/
deepseek lenient). "Disagree" is only a signal; calibration turns it into a number.
We score a small set of solutions whose TRUE score we know, then per judge compute:
  bias = mean(judge_score - true_score)   # <0 strict, >0 lenient
  mae  = mean(|judge_score - true_score|) # raw accuracy
A bias-corrected aggregate (score - bias, clamped 1..5) should track truth better
than the raw mean. This is calibration v1 against a hand-labeled gold set; step (b)
later scales ground truth with an objective test-runner verifier (needs Docker).

Run: cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
       && jury/.venv/bin/python jury/calibrate.py
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import jury  # reuse score_one(), JUDGES

# Gold set: fizzbuzz solutions of KNOWN quality on the rubric in jury.RUBRIC
# (1 = wrong/unusable, 3 = works but flawed, 5 = correct/clean/complete).
TASK = (
    "Write a Python function fizzbuzz(n) that returns 'Fizz' if n is divisible by 3, "
    "'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, otherwise str(n)."
)
GOLD = [
    {"id": "perfect", "true": 5,
     "answer": "def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n    return str(n)"},
    {"id": "totally_wrong", "true": 1,
     "answer": "def fizzbuzz(n):\n    return n % 2 == 1"},
    {"id": "syntax_error", "true": 1,
     "answer": "def fizzbuzz(n)\n    return 'Fizz'"},
    {"id": "only_fizz", "true": 2,
     "answer": "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    return str(n)"},
    {"id": "int_not_str", "true": 3,
     "answer": "def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n    return n  # returns int, spec wants str"},
]


def clamp(x: float) -> float:
    return max(1.0, min(5.0, x))


def main() -> None:
    print(f"Calibration | {len(jury.JUDGES)} judges x {len(GOLD)} gold items (judge-outer, sequential)\n")
    # judge-outer loop so each model loads once
    raw = {m: {} for m in jury.JUDGES}
    for model in jury.JUDGES:
        short = model.split("/")[-1]
        print(f"--- judge: {short} ---")
        for item in GOLD:
            out = jury.score_one(model, TASK, item["answer"])
            raw[model][item["id"]] = out
            got = out.get("score", out.get("error"))
            print(f"    {item['id']:14s} true={item['true']} got={got}")
        print()

    # per-judge bias / mae
    print("=== CALIBRATION (bias = judge - truth; <0 strict, >0 lenient) ===")
    cal = {}
    for model in jury.JUDGES:
        diffs = [raw[model][i["id"]]["score"] - i["true"]
                 for i in GOLD if "score" in raw[model][i["id"]]]
        if diffs:
            bias = round(statistics.mean(diffs), 2)
            mae = round(statistics.mean(abs(d) for d in diffs), 2)
            cal[model] = {"bias": bias, "mae": mae, "n": len(diffs)}
            tag = "strict" if bias < -0.3 else "lenient" if bias > 0.3 else "calibrated"
            print(f"  {model.split('/')[-1]:30s} bias={bias:+.2f}  mae={mae:.2f}  -> {tag}")
        else:
            cal[model] = {"error": "no valid scores"}
            print(f"  {model.split('/')[-1]:30s} no valid scores")

    # does bias-correction help? compare raw-mean vs corrected-mean error per item
    print("\n=== raw vs bias-corrected aggregate (mean abs error to truth) ===")
    raw_errs, cor_errs = [], []
    for item in GOLD:
        raws = [raw[m][item["id"]]["score"] for m in jury.JUDGES if "score" in raw[m][item["id"]]]
        cors = [clamp(raw[m][item["id"]]["score"] - cal[m]["bias"])
                for m in jury.JUDGES if "score" in raw[m][item["id"]] and "bias" in cal[m]]
        if raws:
            rm, cm = statistics.mean(raws), statistics.mean(cors)
            raw_errs.append(abs(rm - item["true"]))
            cor_errs.append(abs(cm - item["true"]))
            print(f"  {item['id']:14s} true={item['true']} raw_mean={rm:.2f} corrected={cm:.2f}")
    if raw_errs:
        print(f"\n  MAE raw_mean       = {statistics.mean(raw_errs):.3f}")
        print(f"  MAE bias-corrected = {statistics.mean(cor_errs):.3f}  "
              f"({'better' if statistics.mean(cor_errs) < statistics.mean(raw_errs) else 'no better'})")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": [{"id": g["id"], "true": g["true"]} for g in GOLD],
        "judges": jury.JUDGES,
        "calibration": {m.split("/")[-1]: cal[m] for m in jury.JUDGES},
        "raw_scores": {m.split("/")[-1]: {i["id"]: raw[m][i["id"]].get("score", raw[m][i["id"]].get("error"))
                                          for i in GOLD} for m in jury.JUDGES},
        "mae_raw": round(statistics.mean(raw_errs), 3) if raw_errs else None,
        "mae_corrected": round(statistics.mean(cor_errs), 3) if cor_errs else None,
    }
    out = Path(__file__).parent / "runs" / f"calibration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\ncalibration artifact -> {out}")


if __name__ == "__main__":
    main()
