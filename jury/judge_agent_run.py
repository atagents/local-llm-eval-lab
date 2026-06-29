"""Close the FreeStack loop: the AGENT solves a real task, the JURY judges its output.

agent (qwen2.5-coder-7b via mini-swe-agent) -> produces solution.py
jury (phi-4 / ministral / llama-3.3, different families, sequential) -> scores it.
Agent model is NEVER one of the judges (agent != judge rule).

The two halves live in different venvs, so the agent runs as a subprocess of its
own venv (mini-swe-agent/.venv) and we judge in this jury venv, reusing score_one().

Run:  cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
        && jury/.venv/bin/python jury/judge_agent_run.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jury  # same dir: reuse score_one(), JUDGES, RUBRIC, SCHEMA

ROOT = Path(__file__).resolve().parents[1]
AGENT_PY = ROOT / "mini-swe-agent" / ".venv" / "bin" / "python"
SMOKE = ROOT / "local_smoke.py"
AGENT_MODEL = os.getenv("AGENT_MODEL", "lm_studio/qwen2.5-coder-7b-instruct")

TASK = (
    "Create a file named solution.py in the current directory containing a single "
    "Python function fizzbuzz(n) that returns 'Fizz' if n is divisible by 3, 'Buzz' "
    "if divisible by 5, 'FizzBuzz' if divisible by both 3 and 5, otherwise str(n). "
    "Then verify the file exists and finish."
)
SOLUTION_FILE = "solution.py"


def run_agent(work_dir: Path) -> dict:
    env = dict(os.environ)  # carries LM_STUDIO_API_BASE/KEY from sourced env-wsl.sh
    env["MSWEA_MODEL_NAME"] = AGENT_MODEL
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [str(AGENT_PY), str(SMOKE), TASK],
        cwd=work_dir, env=env, capture_output=True, text=True, timeout=600,
    )
    return {"rc": proc.returncode, "stdout_tail": proc.stdout[-1200:]}


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_dir = ROOT / "jury" / "runs" / f"agent_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] agent ({AGENT_MODEL.split('/')[-1]}) solving task in {work_dir.name}/ ...")
    agent_out = run_agent(work_dir)
    sol_path = work_dir / SOLUTION_FILE
    if not sol_path.exists():
        print(f"  AGENT DID NOT PRODUCE {SOLUTION_FILE} (rc={agent_out['rc']}).")
        print("  files:", [p.name for p in work_dir.iterdir()])
        print("  agent stdout tail:\n" + agent_out["stdout_tail"])
        sys.exit(1)

    solution = sol_path.read_text()
    print(f"[2/3] agent produced {SOLUTION_FILE} ({len(solution)} chars):")
    print("-" * 50 + "\n" + solution.strip() + "\n" + "-" * 50)

    print(f"[3/3] jury ({len(jury.JUDGES)} judges, sequential) scoring the agent's real output ...")
    results = {}
    for model in jury.JUDGES:
        out = jury.score_one(model, TASK, solution)
        results[model] = out
        short = model.split("/")[-1]
        if "error" in out:
            print(f"    {short:30s} ERROR {out['error']}")
        else:
            print(f"    {short:30s} score={out['score']} ({out['secs']}s)  {out['explanation'][:70]}")

    scores = [r["score"] for r in results.values() if "score" in r]
    verdict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_model": AGENT_MODEL,
        "task": TASK,
        "solution": solution,
        "judges": jury.JUDGES,
        "scores": {m.split("/")[-1]: results[m].get("score", results[m].get("error")) for m in jury.JUDGES},
        "agent_rc": agent_out["rc"],
    }
    agg = jury.aggregate(scores)
    verdict.update(agg)
    if scores:
        flag = "  <-- JUDGES DISAGREE" if agg.get("disagree") else ""
        print(f"\nVERDICT on agent output: mean={agg['mean']} median={agg['median']} "
              f"spread={agg['spread']}{flag}\n  per-judge={verdict['scores']}")

    out_path = work_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(f"\nclosed-loop artifact -> {out_path}")


if __name__ == "__main__":
    main()
