"""PulseSales workflow engine — vendored copy for objective eval in the
Docker --network none verifier sandbox. Source of truth:
pulsesales/backend/app/workflows/engine.py (pure stdlib, no app deps).
"""
from __future__ import annotations

# State machine: { current_state: { trigger: next_state } }
WORKFLOW: dict[str, dict[str, str]] = {
    "new":             {"call_started": "contacting", "skip": "snoozed"},
    "contacting":      {"qualified": "qualified",
                        "not_interested": "lost",
                        "callback": "callback",
                        "voicemail": "voicemail",
                        "no_answer": "no_answer",
                        "error": "error"},
    "no_answer":       {"call_started": "contacting"},
    "voicemail":       {"call_started": "contacting"},
    "callback":        {"call_started": "contacting"},
    "error":           {"call_started": "contacting"},
    "qualified":       {"book_demo": "demo_scheduled",
                        "not_interested": "lost"},
    "demo_scheduled":  {"demo_completed": "demo_completed",
                        "no_show": "contacting",
                        "cancel": "lost"},
    "demo_completed":  {"send_proposal": "proposal_sent",
                        "not_interested": "lost"},
    "proposal_sent":   {"signed": "won",
                        "declined": "lost",
                        "negotiating": "proposal_sent"},
    "won":             {"reopen": "qualified"},
    "lost":            {"reopen": "new"},
    "snoozed":         {"unsnooze": "new"},
}

TERMINAL = {"won", "lost"}


def transition(state: str, trigger: str) -> str:
    """Return the next state, or the same state if the trigger isn't valid."""
    return WORKFLOW.get(state, {}).get(trigger, state)


def trigger_for_outcome(outcome: str | None) -> str:
    """Map a voice agent's call outcome to a workflow trigger."""
    if not outcome:
        return "no_answer"
    o = outcome.lower().strip()
    if o in {"qualified", "not_interested", "callback", "voicemail", "no_answer", "error"}:
        return o
    return "no_answer"
