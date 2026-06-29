"""Measure a BASE model on the metrics that fine-tuning should actually move
(the mean persona-fit is near the ceiling, so it's the wrong target):

  - FAILURE RATE  : % of replies scoring <=2 on persona-fit (the misses fine-tuning should kill)
  - persona-fit   : mean + min + stdev (floor and spread, not just the average)
  - VOICE MATCH   : how close the reply is to the GOLD Mira reference's voice/style (1-5)
  - NO-PROMPT     : persona-fit with NO persona system prompt -> how much it relies on the prompt
                    (fine-tuning should let it hold character with a shorter/empty prompt)

Two-phase (model loaded once, then judge loaded once). $0 local.
Run:  source env-wsl.sh && dashboard/.venv/bin/python companion/eval_base.py \
        --model gemma-4-e4b-it --judge qwen2.5-7b-instruct-1m --dataset companion_hard --n 24
"""
import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import db  # noqa: E402
import evals_lib  # noqa: E402
from seed_companion import PERSONA  # noqa: E402

VOICE_RUBRIC = (
    "Below is a GOLD reply written in Mira's established voice, and a CANDIDATE reply to the same "
    "message. Score 1-5 how well the CANDIDATE matches the GOLD's VOICE and STYLE - tone, warmth, wit, "
    "playfulness, phrasing, rhythm, length - NOT whether it's correct. 5 = sounds like the same person "
    "wrote it. 3 = same idea, different voice. 1 = completely different voice. JSON only.")


def derive_base():
    if os.getenv("LM_STUDIO_API_BASE"):
        return os.getenv("LM_STUDIO_API_BASE")
    try:
        gw = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout.split()[2]
        return f"http://{gw}:1234/v1"
    except Exception:
        return "http://localhost:1234/v1"


def voice_match(judge, api_base, question, candidate, gold):
    user = f"USER said:\n{question}\n\nGOLD (Mira's voice):\n{gold}\n\nCANDIDATE:\n{candidate}"
    d, _, _ = evals_lib._judge(judge, api_base, VOICE_RUBRIC, user, evals_lib.SCORE_SCHEMA)
    return float(d["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--judge", default="qwen2.5-7b-instruct-1m")
    ap.add_argument("--dataset", default="companion_hard")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    base = derive_base()
    cases = db.list_cases(a.dataset)[: a.n]
    if not cases:
        print(f"no cases in '{a.dataset}'"); return

    print(f"\nBASE: {a.model}   JUDGE: {a.judge}   dataset: {a.dataset}   n: {len(cases)}\n")

    # PHASE 1 - generate twice per case (with persona, and with NO prompt). Model loaded once.
    print("PHASE 1 - generate (with-persona + no-prompt) ...")
    gens = []
    for c in cases:
        sysp = c.get("context") or PERSONA
        try:
            g_full = evals_lib.generate(a.model, base, c["input"], system=sysp)
        except Exception as e:  # noqa: BLE001
            g_full = f"[gen error: {str(e)[:80]}]"
        try:
            g_np = evals_lib.generate(a.model, base, c["input"], system="")
        except Exception as e:  # noqa: BLE001
            g_np = f"[gen error: {str(e)[:80]}]"
        gens.append((c, sysp, g_full, g_np))

    # PHASE 2 - judge everything (judge loaded once)
    print(f"PHASE 2 - judge with {a.judge} ...\n")
    pf_full, pf_np, vm, lows = [], [], [], []
    for c, sysp, g_full, g_np in gens:
        s_full = evals_lib.persona_fit(a.judge, base, question=c["input"], answer=g_full, context=sysp)["score"]
        s_np = evals_lib.persona_fit(a.judge, base, question=c["input"], answer=g_np, context=sysp)["score"]
        v = voice_match(a.judge, base, c["input"], g_full, c["output"])
        pf_full.append(s_full); pf_np.append(s_np); vm.append(v)
        if s_full <= 2:
            lows.append((c["input"][:50], s_full, " ".join(g_full.split())[:80]))

    n = len(pf_full)
    fail = sum(1 for s in pf_full if s <= 2) / n
    def mean(xs): return sum(xs) / len(xs)
    def sd(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    print("================= BASE METRICS (what fine-tuning must move) =================")
    print(f"  persona-fit (with prompt) : mean {mean(pf_full):.2f}  min {min(pf_full):.0f}  "
          f"stdev {sd(pf_full):.2f}")
    print(f"  FAILURE RATE (<=2)        : {fail*100:.0f}%  ({sum(1 for s in pf_full if s<=2)}/{n})  "
          f"<- fine-tune should drive this toward 0")
    print(f"  VOICE-MATCH to gold Mira  : mean {mean(vm):.2f}/5  <- 'sounds like MY Mira'")
    print(f"  NO-PROMPT persona-fit     : mean {mean(pf_np):.2f}  (drop vs with-prompt: "
          f"{mean(pf_full)-mean(pf_np):+.2f})  <- lower drop after FT = character baked in")
    print("============================================================================")
    if lows:
        print("\nFailures (<=2) the fine-tune should fix:")
        for inp, sc, snip in lows[:6]:
            print(f"  [{sc:.0f}] {inp!r} -> {snip!r}")


if __name__ == "__main__":
    main()
