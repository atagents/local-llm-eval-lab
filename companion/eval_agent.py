"""Measure the Mira AGENT (Phase F) - not the bare model, the real product.

The agent uses the full Mira persona prompt + (here) stateless per-case isolation, so this asks:
does 'base Qwen + the agent's good prompt' beat the bare-model baseline? Same held-out test
(companion_hard), same metrics (persona-fit, failure-rate, voice-match), same neutral judge.

Two-phase: the agent generates all replies (brain loaded once), then the judge scores all.
Run:  source ./env-wsl.sh && companion/.venv/bin/python companion/eval_agent.py
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT / "companion"))
import db  # noqa: E402
import evals_lib  # noqa: E402
import mira_agent  # noqa: E402
from eval_base import voice_match, derive_base  # noqa: E402

JUDGE = "gemma-4-e4b-it"
N = 24
# bare-model baseline (base Qwen, eval_base, same test + judge) for the comparison:
BASE = {"persona": 3.50, "failure": 0.04, "voice": 2.42}


def main():
    base = derive_base()
    cases = db.list_cases("companion_hard")[:N]
    (ROOT / "companion" / "mira_facts.json").unlink(missing_ok=True)   # clean slate

    print(f"\nAGENT eval | brain {mira_agent.MODEL_ID} | judge {JUDGE} | n {len(cases)}\n")

    # PHASE 1 - the agent answers every case (stateless: fresh history, no memory/tool)
    print("PHASE 1 - Mira agent generates ...")
    gens = []
    for c in cases:
        try:
            text, _ = mira_agent.reply(c["input"], history=[], memory=False)
        except Exception as e:  # noqa: BLE001
            text = f"[agent error: {str(e)[:80]}]"
        gens.append((c, text))

    # PHASE 2 - judge everything (judge loaded once)
    print(f"PHASE 2 - judge with {JUDGE} ...\n")
    pf, vm, lows = [], [], []
    for c, g in gens:
        s = evals_lib.persona_fit(JUDGE, base, question=c["input"], answer=g, context=c.get("context") or "")["score"]
        v = voice_match(JUDGE, base, c["input"], g, c["output"])
        pf.append(s); vm.append(v)
        if s <= 2:
            lows.append((c["input"][:50], s, " ".join(g.split())[:80]))

    n = len(pf)
    fail = sum(1 for s in pf if s <= 2) / n
    def mean(xs): return sum(xs) / len(xs)

    def delta(now, was):
        d = now - was
        return f"({d:+.2f} vs base)"

    print("================= MIRA AGENT vs bare-model baseline =================")
    print(f"  persona-fit  : {mean(pf):.2f}  (base {BASE['persona']:.2f})  {delta(mean(pf), BASE['persona'])}  "
          f"min {min(pf):.0f}  stdev {statistics.pstdev(pf):.2f}")
    print(f"  FAILURE RATE : {fail*100:.0f}%  (base {BASE['failure']*100:.0f}%)")
    print(f"  VOICE-MATCH  : {mean(vm):.2f}  (base {BASE['voice']:.2f})  {delta(mean(vm), BASE['voice'])}")
    print("====================================================================")
    if lows:
        print("\nAgent failures (<=2):")
        for inp, sc, snip in lows[:6]:
            print(f"  [{sc:.0f}] {inp!r} -> {snip!r}")


if __name__ == "__main__":
    main()
