"""Objective verifier for the PulseSales workflow engine.

Two layers:
  1. The repo's own ~25 tests (vendored from pulsesales/backend/tests/test_workflow.py,
     import re-pointed at `solution`).
  2. Lab-added invariants the repo did NOT assert (reachability, terminal sinks,
     retry convergence, no funnel-skip). These are the FreeStack lab earning its keep.

Runs in the Docker --network none sandbox against `solution.py`. Zero model, zero cost.
"""
import pytest
from solution import WORKFLOW, TERMINAL, transition, trigger_for_outcome


# ===========================================================================
# Layer 1 — vendored from the repo's own suite (import re-pointed)
# ===========================================================================

def test_happy_path_new_to_won():
    path = [
        ("new",            "call_started",   "contacting"),
        ("contacting",     "qualified",      "qualified"),
        ("qualified",      "book_demo",      "demo_scheduled"),
        ("demo_scheduled", "demo_completed", "demo_completed"),
        ("demo_completed", "send_proposal",  "proposal_sent"),
        ("proposal_sent",  "signed",         "won"),
    ]
    for state, trigger, expected in path:
        assert transition(state, trigger) == expected


@pytest.mark.parametrize("trigger,expected", [
    ("qualified", "qualified"), ("not_interested", "lost"), ("callback", "callback"),
    ("voicemail", "voicemail"), ("no_answer", "no_answer"), ("error", "error"),
])
def test_contacting_branches(trigger, expected):
    assert transition("contacting", trigger) == expected


@pytest.mark.parametrize("state", ["no_answer", "voicemail", "callback", "error"])
def test_retry_loop_back_to_contacting(state):
    assert transition(state, "call_started") == "contacting"


def test_demo_scheduled_no_show_reengages():
    assert transition("demo_scheduled", "no_show") == "contacting"


def test_demo_scheduled_cancel_loses():
    assert transition("demo_scheduled", "cancel") == "lost"


def test_qualified_not_interested_loses():
    assert transition("qualified", "not_interested") == "lost"


def test_proposal_negotiating_self_loop():
    state = "proposal_sent"
    for _ in range(3):
        state = transition(state, "negotiating")
    assert state == "proposal_sent"


def test_reopen_from_won():
    assert transition("won", "reopen") == "qualified"


def test_reopen_from_lost():
    assert transition("lost", "reopen") == "new"


def test_won_and_lost_are_terminals():
    assert "won" in TERMINAL and "lost" in TERMINAL


def test_snooze_from_new():
    assert transition("new", "skip") == "snoozed"


def test_unsnooze_returns_to_new():
    assert transition("snoozed", "unsnooze") == "new"


def test_unknown_trigger_is_noop():
    assert transition("contacting", "banana") == "contacting"


def test_unknown_state_is_noop():
    assert transition("does_not_exist", "call_started") == "does_not_exist"


def test_trigger_for_outcome_none():
    assert trigger_for_outcome(None) == "no_answer"


def test_trigger_for_outcome_empty_string():
    assert trigger_for_outcome("") == "no_answer"


def test_trigger_for_outcome_gibberish():
    assert trigger_for_outcome("gobbledygook") == "no_answer"


@pytest.mark.parametrize("outcome", [
    "qualified", "not_interested", "callback", "voicemail", "no_answer", "error",
])
def test_trigger_for_outcome_valid_values(outcome):
    assert trigger_for_outcome(outcome) == outcome


def test_trigger_for_outcome_case_insensitive():
    assert trigger_for_outcome("Qualified") == "qualified"
    assert trigger_for_outcome("NOT_INTERESTED") == "not_interested"


def test_trigger_for_outcome_strips_whitespace():
    assert trigger_for_outcome("  voicemail  ") == "voicemail"


def test_all_next_states_are_defined():
    all_keys = set(WORKFLOW.keys())
    for state, transitions in WORKFLOW.items():
        for trigger, next_state in transitions.items():
            assert next_state in all_keys


# ===========================================================================
# Layer 2 — lab-added invariants (NOT in the repo's own suite)
# ===========================================================================

def test_every_state_reachable_from_new():
    """BFS from 'new' must reach every state — no orphan/dead states."""
    seen, frontier = {"new"}, ["new"]
    while frontier:
        s = frontier.pop()
        for nxt in WORKFLOW.get(s, {}).values():
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    missing = set(WORKFLOW.keys()) - seen
    assert not missing, f"unreachable states from 'new': {missing}"


def test_terminals_only_escape_via_reopen():
    for t in TERMINAL:
        assert set(WORKFLOW.get(t, {})) == {"reopen"}, (
            f"terminal {t!r} must only allow 'reopen'"
        )


@pytest.mark.parametrize("state", ["no_answer", "voicemail", "callback", "error"])
def test_retry_branches_only_redial(state):
    """Every retry state must offer exactly one trigger: call_started -> contacting."""
    assert WORKFLOW[state] == {"call_started": "contacting"}


def test_no_funnel_skip():
    """You cannot close the deal without walking the funnel."""
    assert transition("contacting", "signed") == "contacting"
    assert transition("new", "send_proposal") == "new"
    assert transition("qualified", "signed") == "qualified"
    assert transition("new", "signed") == "new"


def test_ai_outcomes_never_advance_funnel():
    """Documents the real gap precisely: the 6 AI-emitted outcomes can only NO-OP
    or drop a lead to 'lost' from deep-funnel states - they NEVER advance it.
    The forward triggers (book_demo/demo_completed/send_proposal/signed) are
    human-only (dashboard advance_lead), so the AI alone cannot push a lead toward
    'won' past 'contacting'. (not_interested CAN lose a qualified/demo_completed lead.)
    """
    ai_outcomes = ["qualified", "not_interested", "callback", "voicemail", "no_answer", "error"]
    for deep_state in ["qualified", "demo_scheduled", "demo_completed", "proposal_sent"]:
        for o in ai_outcomes:
            result = transition(deep_state, trigger_for_outcome(o))
            assert result in {deep_state, "lost"}, (
                f"{deep_state} + AI:{o} -> {result} (AI may only no-op or lose, never advance)"
            )
