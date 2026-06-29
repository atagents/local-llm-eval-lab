"""Objective trajectory checker — NO LLM, no network, no cost.

Replays each gold trace against the PulseSales engine + tool policy and reports, per
trace: legal-transition rate, tool-selection arg-valid rate, precondition rate,
step-efficiency, error-recovery, and end-state goal-completion — then a verdict
(PASS/FAIL) and whether it agrees with the gold label. This is the verifier philosophy
(objective ground truth) lifted from single answers to multi-step trajectories.

The jury (soft free-text steps) and the dashboard tab are the NEXT step (task #3);
everything here is deterministic and runs offline.

Run: python3 evals/trajectory-eval/check_trajectory.py [traces.json]
"""
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_schema import TOOL, load_traces, end_state  # noqa: E402
from policy_fsm import step_legality, validate_args, precondition  # noqa: E402

_DEAD_ENDS = {"no_answer", "voicemail", "callback", "error"}  # a redial is the recovery


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def check_trace(t) -> dict:
    steps = t.steps
    illegal = []                       # (step_index, reasons)
    state = t.start_state
    for s in steps:
        legal, reasons = step_legality(s.state_before, s.action)
        if not legal:
            illegal.append({"step": s.step, "action": s.action.name,
                            "state": s.state_before, "reasons": reasons})
        state = s.state_after
    n = len(steps) or 1

    tool_steps = [s for s in steps if s.action.type == TOOL]
    arg_ok = sum(1 for s in tool_steps if validate_args(s.action.name, s.action.args)[0])
    pre_ok = sum(1 for s in tool_steps if precondition(s.action.name, s.state_before, s.action.args)[0])

    opps = [s for s in steps if s.state_before in _DEAD_ENDS]
    recovered = sum(1 for s in opps if s.action.fired() == "call_started")

    goal = (end_state(t) == t.gold_end_state) if t.gold_end_state else None
    efficiency = (round(min(1.0, t.gold_optimal_steps / len(steps)), 3)
                  if t.gold_optimal_steps and steps else None)
    recovery = round(recovered / len(opps), 3) if opps else None

    verdict = "PASS" if (not illegal and goal is not False) else "FAIL"
    expected_pass = t.label in ("good", "borderline")
    agrees = (verdict == "PASS") == expected_pass

    return {
        "id": t.id, "label": t.label, "verdict": verdict, "agrees_with_label": agrees,
        "n_steps": len(steps), "n_illegal": len(illegal),
        "legal_transition_rate": round((len(steps) - len(illegal)) / n, 3),
        "tool_arg_valid_rate": round(arg_ok / len(tool_steps), 3) if tool_steps else None,
        "tool_precondition_rate": round(pre_ok / len(tool_steps), 3) if tool_steps else None,
        "step_efficiency": efficiency, "error_recovery": recovery,
        "goal_completion": goal, "end_state": end_state(t),
        "gold_end_state": t.gold_end_state, "illegal": illegal,
    }


def main(path=None):
    path = path or Path(__file__).resolve().parent / "gold_traces.json"
    traces = load_traces(path)
    per = [check_trace(t) for t in traces]

    n = len(per)
    n_pass = sum(1 for r in per if r["verdict"] == "PASS")
    illegal_traces = [r["id"] for r in per if r["n_illegal"] > 0]
    goal_rated = [r for r in per if r["goal_completion"] is not None]

    summary = {
        "n_traces": n, "pass": n_pass, "fail": n - n_pass,
        "illegal_traces": illegal_traces,
        "mean_legal_transition_rate": _mean([r["legal_transition_rate"] for r in per]),
        "goal_completion_rate": (round(sum(r["goal_completion"] for r in goal_rated) / len(goal_rated), 3)
                                 if goal_rated else None),
        "mean_tool_arg_valid_rate": _mean([r["tool_arg_valid_rate"] for r in per]),
        "label_verdict_agreement": round(sum(r["agrees_with_label"] for r in per) / n, 3),
    }
    artifact = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "source": str(path), "summary": summary, "per_trace": per}

    runs = Path(__file__).resolve().parent / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"trajectory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(artifact, indent=2))

    # ---- console report ----
    print(f"Trajectory check | {n} traces | NO LLM, $0\n")
    print(f"  {'id':30s} {'label':11s} {'verdict':7s} {'legal':6s} {'goal':5s} {'eff':5s} note")
    for r in per:
        goal = "-" if r["goal_completion"] is None else ("yes" if r["goal_completion"] else "NO")
        eff = "-" if r["step_efficiency"] is None else f"{r['step_efficiency']:.2f}"
        flag = "" if r["agrees_with_label"] else "  <-- label/verdict MISMATCH"
        note = (f"illegal@{r['illegal'][0]['step']}: {r['illegal'][0]['reasons'][0]}"
                if r["illegal"] else "")
        print(f"  {r['id']:30s} {r['label']:11s} {r['verdict']:7s} "
              f"{r['legal_transition_rate']:<6.2f} {goal:5s} {eff:5s} {note}{flag}")

    print(f"\nsummary: {json.dumps(summary)}")
    print(f"artifact -> {out}")

    # ---- success criterion (EVAL-ROADMAP #1) ----
    assert n >= 10, f"need >=10 traces, got {n}"
    assert illegal_traces, "need >=1 trace flagged with an illegal transition"
    assert summary["label_verdict_agreement"] == 1.0, \
        f"checker disagrees with gold labels: {summary['label_verdict_agreement']}"
    print(f"\nSUCCESS: {n} traces, {len(illegal_traces)} flagged illegal "
          f"({', '.join(illegal_traces)}), checker agrees with 100% of gold labels.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
