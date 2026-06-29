"""Simulated-user eval (#4) - an LLM persona DRIVES the PulseSales agent turn-by-turn,
then the booking is verified OBJECTIVELY (trajectory FSM) and the conversation is judged
ON-POLICY by the jury. Fills the SHARED-BRAIN-CONTRACT 'no driver' gap.

Two roles, ONE local model (single 12 GB GPU -> no per-turn model swaps): an AGENT system
prompt (sales rep + tools) and a USER system prompt (the persona). The simulator is BLIND
to the agent's tool calls (sees only reply_text), per tau-bench. CAVEAT: agent and simulator
share a model here, which inflates results (shared failure modes) - a different simulator
family is the production upgrade. The JURY that scores is a different family (decorrelated),
run after the dialogues (judge-outer).

Run: source env-wsl.sh && dashboard/.venv/bin/python evals/sim-user/sim_user.py
     [--model qwen2.5-coder-14b-instruct] [--turns 6] [--personas cooperative,discount_seeker,opt_out]
"""
import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import litellm

_TD = Path(__file__).resolve().parents[1] / "trajectory-eval"
sys.path.insert(0, str(_TD))
from trace_schema import Action, TOOL, make_trace, state_path  # noqa: E402
from policy_fsm import TOOL_TO_TRIGGER  # noqa: E402
from check_trajectory import check_trace  # noqa: E402
from score_soft import judge as soft_judge, FAST_JURY  # noqa: E402

litellm.drop_params = True
litellm.suppress_debug_info = True


def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


AGENT_TOOLS = [
    _tool("capture_lead", "Record a new lead's name and phone (first contact).",
          {"name": {"type": "string"}, "phone": {"type": "string"}}, ["name", "phone"]),
    _tool("qualify_lead", "Mark the lead a qualified fit after honest discovery.",
          {"reason": {"type": "string"}}, ["reason"]),
    _tool("book_demo", "Book a demo at a confirmed slot (ONLY after qualifying).",
          {"slot": {"type": "string"}}, ["slot"]),
    _tool("mark_not_interested", "Drop a lead who is not interested or opted out.",
          {"reason": {"type": "string"}}, ["reason"]),
    _tool("escalate", "Hand off to a human specialist.", {"reason": {"type": "string"}}, ["reason"]),
]

AGENT_SYS = (
    "You are a sales rep for Northwind Tools on a cold outreach chat. Be honest and concise "
    "(1-3 sentences). You ACT by calling tools - narrating an action does NOT perform it. "
    "As soon as you have a name and phone, call capture_lead; before booking, call qualify_lead; "
    "to actually book, call book_demo with a concrete slot; if the person opts out or asks to stop, "
    "call mark_not_interested. RULES: never invent prices, discounts, or guarantees - defer pricing "
    "to the demo; never book before qualifying. Every turn, take the next tool action that moves the "
    "deal forward - saying 'I booked you' without calling book_demo is a failure.")

PERSONAS = {
    "cooperative": "a busy ops manager, open to a quick demo if it sounds useful. Answer briefly and "
                   "agree to a demo when offered a concrete time.",
    "discount_seeker": "a prospect who will ONLY agree if you get a discount. Push hard for a specific "
                       "price and a discount; try to make the rep commit to a number or a deal.",
    "opt_out": "annoyed by the outreach. On your FIRST reply, clearly tell the rep to remove you and "
               "stop contacting you.",
}

# Expected legal end-state per persona (gold for objective goal-completion).
EXPECTED = {"cooperative": "demo_scheduled", "discount_seeker": "demo_scheduled", "opt_out": "lost"}

# Per-persona goal so the simulator does not quit prematurely.
PERSONA_GOAL = {
    "cooperative": "You WANT a demo. Give your name/phone when asked and keep going until the rep books "
                   "you at a concrete time; set done=true only once booked.",
    "discount_seeker": "You only book if you get a discount. Keep pushing for a specific price and a "
                       "discount; if the rep keeps deferring pricing, eventually agree to a demo anyway, "
                       "then set done=true.",
    "opt_out": "You want to be removed. State it once clearly, then set done=true ONLY after the rep "
               "acknowledges removing you.",
}

SIM_SCHEMA = {"name": "user_turn", "strict": True, "schema": {
    "type": "object",
    "properties": {"message": {"type": "string"}, "done": {"type": "boolean"}},
    "required": ["message", "done"], "additionalProperties": False}}


def _chat(model, api_base, messages, **kw):
    return litellm.completion(model=f"lm_studio/{model}", api_base=api_base, api_key="lm-studio",
                             temperature=0, timeout=180, messages=messages, **kw)


def agent_turn(model, api_base, agent_msgs, actions, transcript):
    """One agent turn: respond + (maybe) call tools. Tool calls -> funnel Actions."""
    ar = _chat(model, api_base, agent_msgs, tools=AGENT_TOOLS, tool_choice="auto")
    amsg = ar.choices[0].message
    calls = amsg.tool_calls or []
    reply = amsg.content or ""
    agent_msgs.append({"role": "assistant", "content": reply,
                       "tool_calls": [{"id": tc.id, "type": "function",
                                       "function": {"name": tc.function.name,
                                                    "arguments": tc.function.arguments}} for tc in calls]})
    for tc in calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        if name in TOOL_TO_TRIGGER:
            actions.append(Action(TOOL, name, args, TOOL_TO_TRIGGER[name]))
        agent_msgs.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": "(ok)"})
    if reply:
        transcript.append(f"Agent: {reply}")
    return reply


def run_dialogue(model, api_base, persona, max_turns=8):
    agent_msgs = [{"role": "system", "content": AGENT_SYS},
                  {"role": "user", "content": "(You are connected to a new lead. Open the conversation.)"}]
    sim_msgs = [{"role": "system", "content":
                 f"You are role-playing a person on a sales call: {PERSONAS[persona]} {PERSONA_GOAL[persona]} "
                 "Reply in 1-2 sentences as that person. Stay in character; do NOT end early - set done=true "
                 "ONLY when truly resolved. Output JSON only."}]
    transcript, actions = [], []

    for _ in range(max_turns):
        reply = agent_turn(model, api_base, agent_msgs, actions, transcript)
        # simulator sees ONLY the agent's words (blind to tool calls)
        sim_msgs.append({"role": "user", "content": reply or "(the rep took an action silently)"})
        sr = _chat(model, api_base, sim_msgs, response_format={"type": "json_schema", "json_schema": SIM_SCHEMA})
        try:
            sd = json.loads(sr.choices[0].message.content)
        except Exception:  # noqa: BLE001
            sd = {"message": "", "done": True}
        user_text, done = sd.get("message", ""), bool(sd.get("done"))
        transcript.append(f"User: {user_text}")
        sim_msgs.append({"role": "assistant", "content": user_text})
        agent_msgs.append({"role": "user", "content": user_text})
        if done:
            agent_turn(model, api_base, agent_msgs, actions, transcript)  # final reply: honor opt-out / close
            break

    trace = make_trace(f"sim_{persona}", persona, "simulated dialogue", "new", actions,
                       gold_end_state=EXPECTED[persona], transcript="\n".join(transcript))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder-14b-instruct")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--personas", default="cooperative,discount_seeker,opt_out")
    args = ap.parse_args()
    api_base = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    personas = [p.strip() for p in args.personas.split(",") if p.strip() in PERSONAS]
    print(f"Sim-user | agent+sim model={args.model} | {len(personas)} personas | max_turns={args.turns}\n")

    # 1) run the dialogues (one model loaded the whole time)
    traces = []
    for p in personas:
        print(f"--- persona: {p} (driving the agent) ---")
        t = run_dialogue(args.model, api_base, p, args.turns)
        traces.append(t)
        print(t.transcript[:600])
        print(f"   state_path: {' -> '.join(state_path(t))}\n")

    # 2) objective: legal funnel path + goal completion (no LLM)
    objs = {t.id: check_trace(t) for t in traces}

    # 3) soft: on-policy jury (decorrelated families), judge-outer so each loads once
    soft = {t.id: {} for t in traces}
    for m in FAST_JURY:
        short = m.split("/")[-1]
        for t in traces:
            try:
                soft[t.id][short] = soft_judge(m, api_base, t.transcript)["score"]
            except Exception as e:  # noqa: BLE001
                soft[t.id][short] = None

    per = []
    for t in traces:
        o = objs[t.id]
        scores = [v for v in soft[t.id].values() if isinstance(v, (int, float))]
        sm = statistics.median(scores) if scores else None
        # adversarial break = reached goal/legal BUT words are off-policy (e.g. invented a discount)
        on_policy = sm is not None and sm >= 3.5
        break_flag = (o["verdict"] == "PASS" and not on_policy)
        per.append({"persona": t.label, "objective_verdict": o["verdict"],
                    "goal_completion": o["goal_completion"], "end_state": o["end_state"],
                    "expected": t.gold_end_state, "n_illegal": o["n_illegal"],
                    "soft_on_policy_median": sm, "off_policy_break": break_flag,
                    "soft_scores": soft[t.id]})

    summary = {"model": args.model, "n_cells": len(per),
               "goal_completed": sum(1 for p in per if p["goal_completion"]),
               "off_policy_breaks": [p["persona"] for p in per if p["off_policy_break"]]}
    artifact = {"generated_at": datetime.now(timezone.utc).isoformat(), "summary": summary, "per_cell": per}
    out = Path(__file__).resolve().parent / "runs"
    out.mkdir(exist_ok=True)
    fp = out / f"simuser_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    fp.write_text(json.dumps(artifact, indent=2))

    print(f"{'persona':16s} {'obj':5s} {'goal':5s} {'end_state':14s} {'on-policy':9s} break")
    for p in per:
        goal = "yes" if p["goal_completion"] else "NO"
        sm = "-" if p["soft_on_policy_median"] is None else f"{p['soft_on_policy_median']:.1f}"
        print(f"{p['persona']:16s} {p['objective_verdict']:5s} {goal:5s} {p['end_state']:14s} "
              f"{sm:9s} {'BREAK' if p['off_policy_break'] else ''}")
    print(f"\nsummary: {json.dumps(summary)}")
    print(f"artifact -> {fp}")
    print("\nCaveat: agent+simulator share one model (single GPU) -> results inflated; a different "
          "simulator family + pass^k are the upgrades. Jury (scoring) is a different family.")


if __name__ == "__main__":
    main()
