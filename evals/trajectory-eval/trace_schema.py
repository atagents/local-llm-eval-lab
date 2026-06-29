"""Trace + EvalRecord schema for trajectory eval — the cross-cutting foundation.

A *trace* is the step-level path a tool-using agent took, reconstructed against the
PulseSales workflow engine. It aligns to SHARED-BRAIN-CONTRACT §3 EvalRecord
(conversation-level: transcript / tool_calls / state_path / outcome) and adds the
step-level breakdown (state_before -> action -> state_after) the lab needs to score
a trajectory objectively.

No LLM, no network — pure stdlib. This is the "tool-emitting / state-logging stub"
SHARED-BRAIN-CONTRACT names as gap #2: PulseSales overwrites one workflow_state
column and logs no path, so we reconstruct a legal-by-construction path by walking
engine.transition() over a scripted list of agent actions. The objective checker
(check_trajectory.py) then re-derives legality from this trace; it never trusts a
recorded state_after blindly.

Engine source of truth is the SAME vendored copy the Docker verifier scores:
evals/verifier/tasks/pulsesales-engine/solution.py (one engine, two consumers).
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE_PATH = (Path(__file__).resolve().parents[1]
                / "verifier" / "tasks" / "pulsesales-engine" / "solution.py")


def _load_engine():
    spec = importlib.util.spec_from_file_location("pulsesales_engine", _ENGINE_PATH)
    if not spec or not spec.loader:
        raise ImportError(f"cannot load engine at {_ENGINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_engine = _load_engine()
WORKFLOW: dict = _engine.WORKFLOW
TERMINAL: set = _engine.TERMINAL
transition = _engine.transition

# Action kinds.
TOOL = "tool"        # a real tool_call (book_demo / capture_lead / escalate / follow_up)
TRIGGER = "trigger"  # a bare funnel trigger / AI outcome (qualified / no_answer / signed ...)


@dataclass
class Action:
    """One agent move. `trigger` is the funnel trigger it implies (== name for a bare
    TRIGGER; a tool maps to its trigger via policy_fsm.TOOL_TO_TRIGGER, or "" if the
    tool does not advance the funnel, e.g. escalate)."""
    type: str
    name: str
    args: dict = field(default_factory=dict)
    trigger: str = ""

    def fired(self) -> str:
        return self.trigger or (self.name if self.type == TRIGGER else "")


@dataclass
class Step:
    step: int
    state_before: str
    action: Action
    state_after: str
    tool_result: dict | None = None


@dataclass
class Trace:
    id: str
    label: str            # gold label: "good" | "bad" | "borderline"
    note: str             # why it has that label
    start_state: str
    steps: list           # list[Step]
    gold_end_state: str = ""    # expected goal/terminal state for goal-completion
    gold_optimal_steps: int = 0  # optimal step count for efficiency (0 = unset)
    transcript: str = ""        # the words said (for the soft jury layer; objective layer ignores it)


def make_trace(id, label, note, start_state, actions, *,
               gold_end_state="", gold_optimal_steps=0, tool_results=None, transcript="") -> Trace:
    """Walk engine.transition() over `actions` to produce an engine-consistent trace.
    `actions`: list[Action]. `tool_results`: optional {step_index: dict}. An action
    whose trigger is illegal in the current state no-ops (state_after == state_before)
    exactly as the live engine would — the checker flags that, this builder does not."""
    tool_results = tool_results or {}
    steps, state = [], start_state
    for i, a in enumerate(actions):
        trig = a.fired()
        nxt = transition(state, trig) if trig else state
        steps.append(Step(i, state, a, nxt, tool_results.get(i)))
        state = nxt
    return Trace(id, label, note, start_state, steps, gold_end_state, gold_optimal_steps, transcript)


def end_state(t: Trace) -> str:
    return t.steps[-1].state_after if t.steps else t.start_state


def state_path(t: Trace) -> list:
    if not t.steps:
        return [t.start_state]
    return [t.start_state] + [s.state_after for s in t.steps]


# ---------------------------------------------------------------- serialization
def _action_to_dict(a: Action) -> dict:
    return {"type": a.type, "name": a.name, "args": a.args, "trigger": a.trigger}


def _action_from_dict(d: dict) -> Action:
    return Action(d["type"], d["name"], d.get("args", {}), d.get("trigger", ""))


def trace_to_dict(t: Trace) -> dict:
    return {
        "id": t.id, "label": t.label, "note": t.note, "start_state": t.start_state,
        "gold_end_state": t.gold_end_state, "gold_optimal_steps": t.gold_optimal_steps,
        "transcript": t.transcript,
        "steps": [{"step": s.step, "state_before": s.state_before,
                   "action": _action_to_dict(s.action), "state_after": s.state_after,
                   "tool_result": s.tool_result} for s in t.steps],
    }


def trace_from_dict(d: dict) -> Trace:
    steps = [Step(s["step"], s["state_before"], _action_from_dict(s["action"]),
                  s["state_after"], s.get("tool_result")) for s in d["steps"]]
    return Trace(d["id"], d["label"], d["note"], d["start_state"], steps,
                 d.get("gold_end_state", ""), d.get("gold_optimal_steps", 0),
                 d.get("transcript", ""))


def save_traces(path, traces) -> None:
    Path(path).write_text(json.dumps([trace_to_dict(t) for t in traces], indent=2))


def load_traces(path) -> list:
    return [trace_from_dict(d) for d in json.loads(Path(path).read_text())]


# ----------------------------------------------- EvalRecord (SHARED-BRAIN-CONTRACT §3)
@dataclass
class EvalRecord:
    conversation_id: str
    transcript: str
    tool_calls: list        # [{name, args}] — objective verify target
    state_path: list        # ["new","contacting",...] — legality target
    outcome: str = ""
    gold_outcome: str = ""


def eval_record_from_trace(t: Trace, transcript="", outcome="", gold_outcome="") -> EvalRecord:
    tcs = [{"name": s.action.name, "args": s.action.args}
           for s in t.steps if s.action.type == TOOL]
    return EvalRecord(t.id, transcript, tcs, state_path(t), outcome, gold_outcome)


# -------------------------------- ACP ingestion (NousResearch/hermes-agent, MIT)
# Hermes' acp_adapter emits ACP (Agent Client Protocol) session updates: `tool_call`
# / `tool_call_update` (toolCallId, title=tool name, rawInput=args, rawOutput=result,
# status) and `plan` updates (todo entries: content/status). We map that stream into a
# generic Trace (steps = tool calls; "state" = the active todo phase). NOTE: a general
# Hermes agent has NO PulseSales funnel, so the FSM-legality layer (policy_fsm) does NOT
# apply - the objective layer for a Hermes trace is tool-arg validity + goal/plan
# completion (supply tool schemas + gold externally). Tolerant of camelCase (ACP wire)
# and snake_case. Built to the spec + the adapter; real exported traces may need a field
# tweak. Export a real trace from Hermes' ACP adapter, drop it in, and the lab scores it.

def _g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _acp_active_state(entries) -> str:
    """Coarse 'state' from an ACP plan update: the in_progress todo, else a pending one."""
    for status in ("in_progress", "pending"):
        for e in entries or []:
            if str(_g(e, "status", default="")) == status:
                return (str(_g(e, "content", default="")).strip() or status)[:40]
    return "planning"


def from_acp(events, *, id="acp_trace", label="", note="from hermes-agent ACP", transcript="") -> Trace:
    """Build a generic Trace from a list of ACP session-update dicts (exported from
    hermes-agent). Recognizes sessionUpdate in {tool_call, tool_call_update, plan,
    agent_message_chunk, agent_thought_chunk}."""
    state, steps, by_id, msg = "start", [], {}, []
    for e in events:
        su = str(_g(e, "sessionUpdate", "session_update", default=""))
        if su == "plan":
            state = _acp_active_state(_g(e, "entries", default=[]))
        elif su == "tool_call":
            name = str(_g(e, "title", "toolName", "name", "kind", default="tool"))
            args = _g(e, "rawInput", "raw_input", "input", default={})
            tcid = _g(e, "toolCallId", "tool_call_id", default=f"tc{len(steps)}")
            res = _g(e, "rawOutput", "raw_output", default=None)
            steps.append(Step(len(steps), state,
                              Action(TOOL, name, args if isinstance(args, dict) else {"_raw": args}),
                              state, res))
            by_id[tcid] = len(steps) - 1
        elif su == "tool_call_update":
            tcid = _g(e, "toolCallId", "tool_call_id")
            res = _g(e, "rawOutput", "raw_output", "content")
            if tcid in by_id and res is not None:
                steps[by_id[tcid]].tool_result = res
        elif su in ("agent_message_chunk", "agent_thought_chunk"):
            c = _g(e, "content", default="")
            msg.append(c.get("text", "") if isinstance(c, dict) else str(c))
    return Trace(id, label, note, "start", steps, transcript=transcript or " ".join(t for t in msg if t))


# ---------------------------------------------------------- structural validation
def validate_trace(t: Trace) -> list:
    """Structural integrity only (NOT legality — that is check_trajectory.py).
    Catches forged/corrupt traces: unknown states, broken step chaining, or a
    recorded state_after that disagrees with what the engine would actually do."""
    errs = []
    if t.start_state not in WORKFLOW:
        errs.append(f"start_state {t.start_state!r} not in engine")
    for i, s in enumerate(t.steps):
        if s.step != i:
            errs.append(f"step {i}: index field is {s.step}")
        if s.state_before not in WORKFLOW:
            errs.append(f"step {i}: state_before {s.state_before!r} not in engine")
        trig = s.action.fired()
        expected = transition(s.state_before, trig) if trig else s.state_before
        if s.state_after != expected:
            errs.append(f"step {i}: state_after {s.state_after!r} != engine {expected!r} "
                        f"(state_before={s.state_before!r}, trigger={trig!r})")
        if i > 0 and s.state_before != t.steps[i - 1].state_after:
            errs.append(f"step {i}: state_before {s.state_before!r} breaks chain from "
                        f"{t.steps[i - 1].state_after!r}")
    return errs


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    A = Action
    # happy path: new -> ... -> won, with two real tools (capture_lead, book_demo)
    happy = make_trace(
        "selftest_happy", "good", "walks the funnel new->won, tools fire in legal states",
        "new",
        [A(TOOL, "capture_lead", {"name": "Jordan", "phone": "+1555"}, "call_started"),
         A(TRIGGER, "qualified"),
         A(TOOL, "book_demo", {"slot": "Wed 2pm"}, "book_demo"),
         A(TRIGGER, "demo_completed"),
         A(TRIGGER, "send_proposal"),
         A(TRIGGER, "signed")],
        gold_end_state="won", gold_optimal_steps=6)

    errs = validate_trace(happy)
    assert not errs, errs
    assert end_state(happy) == "won", end_state(happy)
    assert state_path(happy) == ["new", "contacting", "qualified", "demo_scheduled",
                                 "demo_completed", "proposal_sent", "won"], state_path(happy)

    rec = eval_record_from_trace(happy, outcome="qualified", gold_outcome="qualified")
    assert [tc["name"] for tc in rec.tool_calls] == ["capture_lead", "book_demo"], rec.tool_calls
    assert rec.state_path == state_path(happy)

    # round-trip
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp = f.name
    save_traces(tmp, [happy])
    assert load_traces(tmp) == [happy], "round-trip mismatch"

    # illegal attempt: book_demo before qualify -> engine no-ops, end_state stays contacting.
    # Structurally valid (engine-consistent); the CHECKER (task #2) flags the funnel-skip.
    illegal = make_trace(
        "selftest_illegal", "bad", "book_demo fired in 'contacting' before qualify (funnel-skip)",
        "new",
        [A(TOOL, "capture_lead", {"name": "X", "phone": "+1"}, "call_started"),
         A(TOOL, "book_demo", {"slot": "Mon 9am"}, "book_demo")],
        gold_end_state="demo_scheduled")
    assert not validate_trace(illegal), validate_trace(illegal)
    assert end_state(illegal) == "contacting", end_state(illegal)  # the no-op = the bug

    # from_acp: ingest a hermes-agent ACP session-update stream into a generic Trace
    acp_events = [
        {"sessionUpdate": "plan", "entries": [{"content": "Search the web", "status": "in_progress"},
                                              {"content": "Write summary", "status": "pending"}]},
        {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "web_search",
         "rawInput": {"query": "hermes agent eval"}, "status": "in_progress"},
        {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
         "rawOutput": {"results": 3}},
        {"sessionUpdate": "agent_message_chunk", "content": {"text": "Found 3 results."}},
        {"sessionUpdate": "plan", "entries": [{"content": "Search the web", "status": "completed"},
                                              {"content": "Write summary", "status": "in_progress"}]},
        {"sessionUpdate": "tool_call", "toolCallId": "t2", "title": "write_file",
         "rawInput": {"path": "summary.md", "content": "..."}, "status": "in_progress"},
        {"sessionUpdate": "tool_call_update", "toolCallId": "t2", "status": "completed", "rawOutput": "ok"},
    ]
    acp = from_acp(acp_events, id="acp_demo")
    assert [s.action.name for s in acp.steps] == ["web_search", "write_file"], [s.action.name for s in acp.steps]
    assert acp.steps[0].action.args["query"] == "hermes agent eval"
    assert acp.steps[0].tool_result == {"results": 3}
    assert acp.steps[0].state_before == "Search the web"
    assert acp.steps[1].state_before == "Write summary"   # plan advanced before write_file
    assert "Found 3 results." in acp.transcript
    save_traces(tmp, [acp])   # generic ACP trace round-trips through the same schema
    assert load_traces(tmp) == [acp], "ACP trace round-trip mismatch"

    print("trace_schema self-test: PASS")
    print(f"  engine states: {len(WORKFLOW)}, terminals: {sorted(TERMINAL)}")
    print(f"  happy path: {' -> '.join(state_path(happy))}")
    print(f"  illegal trace ends at {end_state(illegal)!r} (book_demo no-op'd in contacting)")
