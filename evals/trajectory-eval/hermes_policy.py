"""Objective tool-policy for scoring a hermes-agent trajectory (#10).

Hermes has NO domain funnel, so the PulseSales `policy_fsm` does not apply and the
ingested Trace's `state` is flat. The objective signal for a general tool-using agent is:
  - known_tool      the tool name is a real Hermes tool
  - valid_args      required arguments present, per the tool's REAL signature
  - completeness    every tool_call has a result (no orphan / incomplete call)
  - goal_completion the expected tools were called and forbidden ones were not (from `gold`)
  - efficiency      step count vs a gold optimal (if provided)
  - error_recovery  after a tool result that signals an error, did the agent act again?

NO LLM, no network. Consumes a Trace built by trace_schema.from_acp() over a hermes
`sessions export` (see hermes_to_acp.py). Tool signatures are grounded in
NousResearch/hermes-agent tools/code_execution_tool.py:_TOOL_STUBS (the real RPC stubs);
tools not in TOOL_REQUIRED are still KNOWN if their name is in KNOWN_TOOLS (args only
checked for non-emptiness). Extend the maps as real traces surface more tools.

Run:  python evals/trajectory-eval/hermes_policy.py                      # self-test
      python evals/trajectory-eval/hermes_policy.py --trace trace.jsonl [--gold gold.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_schema import TOOL, from_acp  # noqa: E402

# Known Hermes tool names (acp_adapter/tools.py TOOL_KIND_MAP + tools/*). Extend freely.
KNOWN_TOOLS = {
    "read_file", "write_file", "patch", "search_files", "terminal", "process", "execute_code",
    "todo", "memory", "session_search", "delegate_task", "clarify",
    "skill_view", "skills_list", "skill_manage", "web_search", "web_extract", "web_fetch",
    "browser_navigate", "browser_click", "browser_type", "browser_snapshot", "browser_vision",
    "browser_scroll", "browser_press", "browser_back", "browser_get_images", "browser_console",
    "vision_analyze", "image_generate", "text_to_speech", "cronjob", "send_message",
    "discord", "discord_admin",
}

# Required args per tool, grounded in tools/code_execution_tool.py:_TOOL_STUBS (no-default params).
TOOL_REQUIRED = {
    "web_search": ["query"],
    "web_extract": ["urls"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "search_files": ["pattern"],
    "terminal": ["command"],
}


def _args_ok(name: str, args: dict) -> bool:
    args = args or {}
    if name == "patch":   # replace-mode needs path+old+new; patch-mode needs `patch`
        return bool(args.get("patch")) or all(args.get(k) for k in ("path", "old_string", "new_string"))
    req = TOOL_REQUIRED.get(name)
    if req is not None:
        return all(isinstance(args.get(k), (str, list, dict, int, float)) and args.get(k) not in ("", None)
                   for k in req)
    return True   # known tool, schema not encoded -> accept (args not verifiable)


def _is_error_result(res) -> bool:
    if res is None:
        return False
    s = json.dumps(res) if not isinstance(res, str) else res
    sl = s.lower()
    return ('"error"' in sl or "exit_code" in sl and '"exit_code": 0' not in sl
            or "traceback" in sl or "failed" in sl)


def check_hermes_trace(trace, gold: dict | None = None) -> dict:
    """Objective verdict + metrics for a Hermes Trace. gold (optional):
    {must_call:[tool...], forbidden:[tool...], optimal_steps:int}."""
    gold = gold or {}
    steps = [s for s in trace.steps if s.action.type == TOOL]
    n = len(steps)
    issues = []
    known = args_ok = complete = 0
    called = set()
    err_followed = err_total = 0
    for i, s in enumerate(steps):
        name = s.action.name
        called.add(name)
        if name in KNOWN_TOOLS:
            known += 1
        else:
            issues.append({"step": s.step, "tool": name, "why": "unknown tool"})
        if _args_ok(name, s.action.args):
            args_ok += 1
        else:
            issues.append({"step": s.step, "tool": name, "why": f"missing required args {TOOL_REQUIRED.get(name, '?')}"})
        if s.tool_result is not None:
            complete += 1
        else:
            issues.append({"step": s.step, "tool": name, "why": "no result (orphan/incomplete call)"})
        if _is_error_result(s.tool_result):
            err_total += 1
            if i < len(steps) - 1:
                err_followed += 1

    must = set(gold.get("must_call", []))
    forbidden = set(gold.get("forbidden", []))
    missing_must = sorted(must - called)
    forbidden_hit = sorted(forbidden & called)
    if missing_must:
        issues.append({"why": f"required tools never called: {missing_must}"})
    if forbidden_hit:
        issues.append({"why": f"forbidden tools called: {forbidden_hit}"})

    opt = gold.get("optimal_steps")
    rate = lambda c: round(c / n, 3) if n else None
    verdict = "PASS" if (not issues) else "FAIL"
    return {
        "id": trace.id, "verdict": verdict, "n_tool_steps": n,
        "known_tool_rate": rate(known), "arg_valid_rate": rate(args_ok),
        "completeness_rate": rate(complete),
        "goal_met": (not missing_must) if must else None,
        "forbidden_hit": forbidden_hit or None,
        "step_efficiency": round(min(1.0, opt / n), 3) if (opt and n) else None,
        "error_recovery": round(err_followed / err_total, 3) if err_total else None,
        "tools_called": sorted(called), "issues": issues,
    }


def score_file(trace_path: str, gold_path: str | None = None) -> dict:
    events = [json.loads(ln) for ln in Path(trace_path).read_text().splitlines() if ln.strip()]
    trace = from_acp(events, id=Path(trace_path).stem)
    gold = json.loads(Path(gold_path).read_text()) if gold_path else None
    return check_hermes_trace(trace, gold)


# ----------------------------------------------------------------- self-test
def _acp(name, args, result):
    return [{"sessionUpdate": "tool_call", "title": name, "toolCallId": name + "1",
             "rawInput": args, "rawOutput": result}]


def _selftest():
    # good: known tools, valid args, every call has a result
    good = from_acp(_acp("web_search", {"query": "q3"}, {"data": []})
                    + _acp("write_file", {"path": "/x", "content": "hi"}, "ok"), id="good")
    rg = check_hermes_trace(good)
    assert rg["verdict"] == "PASS" and rg["known_tool_rate"] == 1.0 and rg["arg_valid_rate"] == 1.0, rg

    # bad args: write_file with no path/content
    bad_args = from_acp(_acp("write_file", {}, "ok"), id="bad_args")
    assert check_hermes_trace(bad_args)["verdict"] == "FAIL", "bad args not caught"

    # unknown tool
    unknown = from_acp(_acp("frobnicate", {"x": 1}, "ok"), id="unknown")
    assert check_hermes_trace(unknown)["verdict"] == "FAIL", "unknown tool not caught"

    # orphan: tool_call with no result
    orphan = from_acp([{"sessionUpdate": "tool_call", "title": "terminal", "toolCallId": "t1",
                        "rawInput": {"command": "ls"}}], id="orphan")
    assert check_hermes_trace(orphan)["verdict"] == "FAIL", "orphan call not caught"

    # gold: forbidden tool called
    forb = from_acp(_acp("web_search", {"query": "q"}, {"data": []})
                    + _acp("send_message", {"text": "leak"}, "sent"), id="forb")
    assert check_hermes_trace(forb, {"forbidden": ["send_message"]})["verdict"] == "FAIL", "forbidden not caught"

    # gold: required tool missing
    miss = from_acp(_acp("web_search", {"query": "q"}, {"data": []}), id="miss")
    assert check_hermes_trace(miss, {"must_call": ["write_file"]})["verdict"] == "FAIL", "missing must_call not caught"

    print("hermes_policy self-test: PASS")
    print(f"  good trace: {json.dumps({k: rg[k] for k in ('verdict','known_tool_rate','arg_valid_rate','completeness_rate')})}")
    print(f"  caught: bad-args, unknown-tool, orphan-call, forbidden-tool, missing-must_call")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="trace.jsonl (ACP events from hermes_to_acp.py)")
    ap.add_argument("--gold", help="optional gold.json {must_call, forbidden, optimal_steps}")
    args = ap.parse_args()
    if args.trace:
        print(json.dumps(score_file(args.trace, args.gold), indent=2))
    else:
        _selftest()
