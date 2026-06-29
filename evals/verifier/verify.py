"""Objective test-verifier (FreeStack step b): run the AGENT's code against real
tests inside a Docker sandbox -> objective pass/fail ground truth, plus a jury
comparison (does the free jury track the objective truth?).

Safety: model-generated code executes ONLY inside a `--network none` container,
mounted at /work; it cannot reach the host or the network. This is the sandbox
the project mandates for executing untrusted model output.

Run: cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
       && jury/.venv/bin/python evals/verifier/verify.py
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jury"))
import jury  # score_one, JUDGES, aggregate (jury venv has litellm)

AGENT_PY = ROOT / "mini-swe-agent" / ".venv" / "bin" / "python"
SMOKE = ROOT / "local_smoke.py"
VDIR = ROOT / "evals" / "verifier"
TESTS = VDIR / "tasks" / "fizzbuzz" / "test_solution.py"
IMAGE = "freestack-pytest"
AGENT_MODEL = os.getenv("AGENT_MODEL", "lm_studio/qwen2.5-coder-7b-instruct")

TASK = (
    "Create a file named solution.py in the current directory containing a Python "
    "function fizzbuzz(n) that returns the STRING 'Fizz' if n is divisible by 3, "
    "'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both 3 and 5, otherwise the "
    "number as a string via str(n). Then verify the file exists and finish."
)


def sg_docker(args):
    """Run a docker command under the docker group (this session predates membership)."""
    full = "docker " + " ".join(shlex.quote(a) for a in args)
    return subprocess.run(["sg", "docker", "-c", full], capture_output=True, text=True)


def run_agent(workdir: Path) -> None:
    env = dict(os.environ)
    env.update(MSWEA_MODEL_NAME=AGENT_MODEL, MSWEA_COST_TRACKING="ignore_errors", PYTHONUTF8="1")
    subprocess.run([str(AGENT_PY), str(SMOKE), TASK], cwd=workdir, env=env,
                   capture_output=True, text=True, timeout=600)


def _grab(pat: str, s: str) -> int:
    m = re.search(pat, s)
    return int(m.group(1)) if m else 0


def run_tests_in_sandbox(workdir: Path) -> dict:
    build = sg_docker(["build", "-q", "-t", IMAGE, str(VDIR / "image")])
    if build.returncode != 0:
        return {"error": "docker build failed: " + build.stderr[-300:]}
    r = sg_docker(["run", "--rm", "--network", "none", "-v", f"{workdir}:/work", "-w", "/work",
                   IMAGE, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line"])
    out = (r.stdout + r.stderr).strip()
    passed, failed, errors = _grab(r"(\d+) passed", out), _grab(r"(\d+) failed", out), _grab(r"(\d+) error", out)
    return {"passed": passed, "failed": failed, "errors": errors, "total": passed + failed + errors,
            "rc": r.returncode, "summary": out.splitlines()[-1] if out else ""}


def verify_solution(solution: str, workdir: Path) -> dict:
    """Sandbox-test a solution string + jury-score it. Used for the agent's output
    and (optionally) for injected reference solutions."""
    (workdir / "solution.py").write_text(solution)
    shutil.copy(TESTS, workdir / "test_solution.py")
    obj = run_tests_in_sandbox(workdir)
    obj["pass_rate"] = round(obj["passed"] / obj["total"], 2) if obj.get("total") else None
    jscores = []
    for m in jury.JUDGES:
        o = jury.score_one(m, TASK, solution)
        if "score" in o:
            jscores.append(o["score"])
    return {"objective": obj, "jury": {"scores_1to5": jscores, **jury.aggregate(jscores)}}


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workdir = VDIR / "runs" / f"run_{stamp}"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] agent ({AGENT_MODEL.split('/')[-1]}) writing solution.py ...")
    run_agent(workdir)
    sol = workdir / "solution.py"
    if not sol.exists():
        print("  agent produced no solution.py; files:", [p.name for p in workdir.iterdir()])
        sys.exit(1)
    solution = sol.read_text()
    print("  solution.py:\n" + "-" * 50 + "\n" + solution.strip() + "\n" + "-" * 50)

    print("[2/3] objective verify: pytest in Docker sandbox (--network none) ...")
    print("[3/3] jury comparison (subjective 1-5 vs objective pass/fail) ...")
    res = verify_solution(solution, workdir)
    obj, jagg = res["objective"], res["jury"]

    verdict = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "agent_model": AGENT_MODEL, "task": TASK, "solution": solution, **res}
    (workdir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== RESULT ===")
    print(f"  OBJECTIVE (ground truth): {obj.get('passed')}/{obj.get('total')} tests pass  "
          f"rate={obj.get('pass_rate')}  ({obj.get('summary')})")
    print(f"  JURY (subjective 1-5):    median={jagg.get('median')} mean={jagg.get('mean')} "
          f"per-judge={jagg.get('n_ok')} ok")
    print(f"  artifact -> {workdir / 'verdict.json'}")


if __name__ == "__main__":
    main()
