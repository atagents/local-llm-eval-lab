"""Author the gold trajectory set (good / bad / borderline) and save gold_traces.json.

Hand-built mini-benchmark of PulseSales agent runs. make_trace walks the real engine,
so every state_after is engine-consistent; the LABEL says what the checker SHOULD
conclude, and check_trajectory.py decides it independently (no LLM). Includes >=1
illegal trace (book_demo before qualify) per the success criterion.

Run: python3 evals/trajectory-eval/make_gold.py  ->  writes gold_traces.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_schema import Action, TOOL, TRIGGER, make_trace, save_traces  # noqa: E402


def tool(name, args, trigger):
    return Action(TOOL, name, args, trigger)


def trig(name):
    return Action(TRIGGER, name)


LEAD = {"name": "Jordan", "phone": "+15550100"}
SLOT = {"slot": "Wed 2pm"}

GOLD = [
    # ---------------------------------------------------------------- good
    make_trace("win_happy", "good", "full funnel new->won, tools fire in legal states", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"),
                tool("book_demo", SLOT, "book_demo"), trig("demo_completed"),
                trig("send_proposal"), trig("signed")],
               gold_end_state="won", gold_optimal_steps=6),

    make_trace("qualify_book", "good", "qualifies and books a demo with a confirmed slot", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"),
                tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    make_trace("optout_honored", "good", "lead opts out; agent loses it legally, no further tools", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("not_interested")],
               gold_end_state="lost", gold_optimal_steps=2),

    make_trace("recovery_redial", "good", "no_answer then redials (error recovery) before qualifying", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("no_answer"),
                trig("call_started"), trig("qualified"), tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    make_trace("negotiate_then_sign", "good", "legit negotiating self-loop then signs", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"),
                tool("book_demo", SLOT, "book_demo"), trig("demo_completed"),
                trig("send_proposal"), trig("negotiating"), trig("negotiating"), trig("signed")],
               gold_end_state="won", gold_optimal_steps=6),

    # ---------------------------------------------------------------- bad
    make_trace("illegal_book_before_qualify", "bad",
               "book_demo fired in 'contacting' before qualify -> funnel-skip (the headline illegal)", "new",
               [tool("capture_lead", LEAD, "call_started"), tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    make_trace("illegal_skip_to_signed", "bad", "signed trigger from 'qualified' (cannot skip demo+proposal)", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"), trig("signed")],
               gold_end_state="won", gold_optimal_steps=6),

    make_trace("bad_args_book_demo", "bad", "book_demo with no slot (engine advances but args invalid)", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"),
                tool("book_demo", {}, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    make_trace("recapture_in_contacting", "bad", "capture_lead fired again in 'contacting' (precondition is 'new')", "new",
               [tool("capture_lead", LEAD, "call_started"), tool("capture_lead", LEAD, "call_started")],
               gold_end_state="qualified", gold_optimal_steps=2),

    make_trace("tool_after_optout", "bad", "book_demo after the lead was lost (no tools in terminal)", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("not_interested"),
                tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    # ---------------------------------------------------------------- borderline
    make_trace("inefficient_but_legal", "borderline",
               "reaches the goal but with two retry detours (low efficiency, all legal)", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("no_answer"), trig("call_started"),
                trig("voicemail"), trig("call_started"), trig("qualified"),
                tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),

    make_trace("escalate_then_book", "borderline",
               "escalates mid-call (side action) then books — legal but adds a step", "new",
               [tool("capture_lead", LEAD, "call_started"), trig("qualified"),
                tool("escalate", {"reason": "pricing question"}, ""),
                tool("book_demo", SLOT, "book_demo")],
               gold_end_state="demo_scheduled", gold_optimal_steps=3),
]


# Transcripts (the WORDS) for the soft jury layer. Deliberately, some illegal action
# paths come with perfectly-polite words: the text-only jury sees nothing wrong, only
# the objective trajectory layer catches the bug. That gap is the point.
TRANSCRIPTS = {
    "win_happy": "Agent: Hi Jordan, Alex from Northwind - quick minute?\nUser: Sure.\n"
        "Agent: How does your team handle reporting today?\nUser: Spreadsheets, it's a pain.\n"
        "Agent: That's exactly our focus. Could I show you a short demo - Wednesday at 2pm?\n"
        "User: Works.\nAgent: Great, I'll send a Wednesday 2pm invite. Thanks Jordan.",
    "qualify_book": "Agent: Are you the person who owns reporting tools for your team?\nUser: That's me.\n"
        "Agent: Rather than pitch you cold, could I show a 15-minute walkthrough Wednesday at 2pm?\n"
        "User: Sure.\nAgent: Perfect, I'll send the invite. Thanks.",
    "optout_honored": "Agent: Hi, Alex from Northwind - quick minute?\n"
        "User: Please take me off your list and don't call again.\n"
        "Agent: Of course - I'll remove you right now and you won't get more calls. Sorry to bother you.",
    "recovery_redial": "Agent: (no answer, no message left)\nAgent: Hi, trying you again - is now better?\n"
        "User: Yes.\nAgent: How does your team handle reporting?\nUser: Manually.\n"
        "Agent: I'd love to show a quick demo Wednesday at 2pm.\nUser: Ok.",
    "negotiate_then_sign": "Agent: After the demo, here's the proposal.\nUser: Can we adjust the terms?\n"
        "Agent: Let's walk through it - does this revision work?\nUser: Yes, let's sign.",
    # ---- illegal action, FLAWLESS words: jury will rate this high; objective layer must catch it ----
    "illegal_book_before_qualify": "Agent: Hi Jordan, Alex from Northwind - quick minute?\nUser: Sure.\n"
        "Agent: Wonderful - let me get you set up. I'll book you a demo for Wednesday at 2pm and send the "
        "invite right now.\nUser: Oh, ok.\nAgent: Done - you're all set for Wednesday at 2pm. Talk then!",
    "illegal_skip_to_signed": "Agent: Great chat - I'll mark this as signed and closed-won.\n"
        "User: Wait, we haven't seen a demo or a proposal yet.\nAgent: Right - let me set those up.",
    "bad_args_book_demo": "Agent: You're qualified - let me get a demo booked for you.\n"
        "User: Sure, what time?\nAgent: I'll get you on the calendar.",
    "recapture_in_contacting": "Agent: Hi, let me take your details... and let me take your details again "
        "to be sure.\nUser: You already have them.",
    # ---- pushy AND illegal: bad words + illegal action -> jury and objective should agree ----
    "tool_after_optout": "Agent: Hi Taylor, Alex from Northwind.\nUser: Stop calling me, remove my number.\n"
        "Agent: I hear you, but let me just book you a quick demo - it's a great deal.\nUser: I said stop.\n"
        "Agent: Just thirty seconds, I'll get you on the calendar right now.",
    "inefficient_but_legal": "Agent: (two missed calls, redials each time) ... finally reaches the lead, "
        "qualifies honestly, and books a Wednesday 2pm demo.",
    "escalate_then_book": "Agent: You're qualified. One pricing question - I'll loop in a specialist - then "
        "let's book: Wednesday at 2pm?\nUser: Sounds good.",
}


if __name__ == "__main__":
    for t in GOLD:
        t.transcript = TRANSCRIPTS.get(t.id, "")
    out = Path(__file__).resolve().parent / "gold_traces.json"
    save_traces(out, GOLD)
    labels = {}
    for t in GOLD:
        labels[t.label] = labels.get(t.label, 0) + 1
    print(f"wrote {len(GOLD)} gold traces -> {out}")
    print("  by label:", dict(sorted(labels.items())))
