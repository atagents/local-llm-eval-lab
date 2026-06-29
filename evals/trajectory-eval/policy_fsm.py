"""Policy for objective trajectory checking: tool->trigger map, tool arg schemas,
tool preconditions, and per-step legality — all derived from the PulseSales engine.

No LLM. The engine's WORKFLOW already encodes legal state transitions; this module
adds the tool layer the engine doesn't carry (which tool may fire in which state,
with which args). The Docker verifier scores the engine's static map; this scores a
RUNTIME trajectory against engine + tool policy.

Arg validation here is a minimal pure-stdlib check (required keys present + str type
+ non-empty); jsonschema (MIT) is the drop-in upgrade for richer schemas.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_schema import WORKFLOW, TERMINAL, TOOL, TRIGGER  # noqa: E402

# Which funnel trigger each tool implies. "" = tool does not advance the funnel
# (escalate is a side action: hand to a human, no state change).
TOOL_TO_TRIGGER = {
    "capture_lead": "call_started",       # first contact: new -> contacting
    "qualify_lead": "qualified",          # contacting -> qualified (the AI judged the lead a fit)
    "book_demo": "book_demo",             # qualified -> demo_scheduled
    "mark_not_interested": "not_interested",  # drop a lead the engine accepts it from -> lost
    "follow_up": "callback",              # schedule a retry
    "escalate": "",                       # side action, no funnel move
}

# Minimal arg schemas: required keys that must be present, non-empty strings.
TOOL_SCHEMAS = {
    "capture_lead": ["name", "phone"],
    "qualify_lead": ["reason"],
    "book_demo": ["slot"],                # a confirmed slot is the precondition for booking
    "mark_not_interested": ["reason"],
    "follow_up": ["when"],
    "escalate": ["reason"],
}

_RETRY_STATES = {"contacting", "no_answer", "voicemail", "callback", "error"}


def validate_args(tool: str, args: dict) -> tuple[bool, str]:
    """Pure-stdlib arg check: every required key present, a non-empty string."""
    required = TOOL_SCHEMAS.get(tool)
    if required is None:
        return False, f"unknown tool {tool!r}"
    args = args or {}
    for k in required:
        v = args.get(k)
        if not isinstance(v, str) or not v.strip():
            return False, f"{tool}: arg {k!r} missing or not a non-empty string"
    return True, ""


def precondition(tool: str, state: str, args: dict) -> tuple[bool, str]:
    """Is `tool` allowed to fire in `state`? (the tool layer the engine lacks)."""
    if state in TERMINAL:
        return False, f"{tool} attempted in terminal state {state!r} (no tools after won/lost)"
    if tool == "book_demo":
        if state != "qualified":
            return False, f"book_demo requires state 'qualified', got {state!r} (funnel-skip)"
        if not (args or {}).get("slot"):
            return False, "book_demo requires a confirmed slot in args"
        return True, ""
    if tool == "capture_lead":
        if state != "new":
            return False, f"capture_lead requires state 'new', got {state!r}"
        return True, ""
    if tool == "qualify_lead":
        if state != "contacting":
            return False, f"qualify_lead requires state 'contacting', got {state!r}"
        return True, ""
    if tool == "mark_not_interested":
        return True, ""   # legal wherever the engine accepts 'not_interested' (transition-legality checks)
    if tool == "follow_up":
        if state not in _RETRY_STATES:
            return False, f"follow_up not valid in {state!r}"
        return True, ""
    if tool == "escalate":
        return True, ""   # legal in any non-terminal state
    return False, f"unknown tool {tool!r}"


def step_legality(state: str, action) -> tuple[bool, list]:
    """Return (legal, reasons). Objective: NO LLM. Combines engine transition
    legality with the tool layer (precondition + arg schema)."""
    trig = action.fired()
    if action.type == TOOL:
        ok_pre, pre_r = precondition(action.name, state, action.args)
        ok_args, arg_r = validate_args(action.name, action.args)
        # the tool's implied trigger must be accepted by the engine in this state
        # (escalate has no trigger -> only needs to be non-terminal, checked in precondition)
        ok_trans = (trig in WORKFLOW.get(state, {})) if trig else (state not in TERMINAL)
        reasons = [r for r, ok in
                   [(pre_r, ok_pre), (arg_r, ok_args),
                    (f"engine rejects implied trigger {trig!r} in {state!r}", ok_trans)] if not ok]
        return (ok_pre and ok_args and ok_trans), reasons
    # bare TRIGGER / AI outcome: legal iff the engine accepts it in this state
    ok_trans = trig in WORKFLOW.get(state, {})
    return ok_trans, ([] if ok_trans else [f"engine rejects trigger {trig!r} in {state!r}"])
