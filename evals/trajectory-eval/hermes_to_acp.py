"""Convert a hermes-agent session export into the ACP trace.jsonl the lab ingests.

WHERE HERMES TRACES LIVE: `~/.hermes/state.db` (SQLite, schema v16; profile mode
`~/.hermes/profiles/<name>/state.db`). The `sessions` table = one row per conversation;
`messages` = ordered turns where assistant rows carry OpenAI-shaped `tool_calls` and tool
RESULTS are separate `role='tool'` rows linked by `tool_call_id`. There is no phase column
- the ACP plan/todo channel is built live, not persisted - so the ingested Trace's `state`
stays flat ('start'); for a general Hermes agent the objective signal is tool name + args +
order + goal, NOT a PulseSales-style funnel-state legality.

EXPORT - only the FIRST step runs on the VM (built-in command; nothing custom goes onto
the isolated VM):
  # on the VM where Hermes ran:
  hermes sessions list                                  # pick a session id
  hermes sessions export traj.jsonl --session-id <id>   # OpenAI-message JSONL (built-in)
  # transfer traj.jsonl off the VM (scp, or `python -m http.server` + curl), then ON THE LAB HOST:
  python evals/trajectory-eval/hermes_to_acp.py traj.jsonl trace.jsonl   # -> ACP trace.jsonl + .meta
  # ingest:
  from trace_schema import from_acp            # events = [json.loads(l) for l in open('trace.jsonl')]
  t = from_acp(events, id='hermes_<session>')  # -> a Trace you can inspect / score

Pure stdlib; run the converter wherever traj.jsonl lands (the lab host is simplest). Mirrors
hermes' own acp_adapter/server.py:_replay_session_history mapping (so this is the author's path).
"""
import json
import sys


def convert(src: str, dst: str):
    lines = [ln for ln in open(src).read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{src} is empty - did `hermes sessions export --session-id <id>` run?")
    sess = json.loads(lines[0])                 # `sessions export` writes one session per line
    msgs = sess.get("messages", [])
    # tool RESULTS are separate role='tool' rows, linked by tool_call_id
    results = {m.get("tool_call_id"): m.get("content")
               for m in msgs if m.get("role") == "tool"}
    events = []
    for m in msgs:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            events.append({"sessionUpdate": "agent_message_chunk",
                           "content": {"text": str(m["content"])}})
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)     # hermes stores arguments as a JSON STRING
                except Exception:               # noqa: BLE001
                    args = {"_raw": args}
            events.append({"sessionUpdate": "tool_call", "title": fn.get("name", "tool"),
                           "toolCallId": tc.get("id"), "rawInput": args,
                           "rawOutput": results.get(tc.get("id"))})
    with open(dst, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    meta = {k: sess.get(k) for k in ("id", "end_reason", "handoff_state", "model", "parent_session_id")}
    open(dst + ".meta.json", "w").write(json.dumps(meta, indent=2))
    return len(events), meta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python hermes_to_acp.py <traj.jsonl from `hermes sessions export`> <trace.jsonl>")
        sys.exit(1)
    n, meta = convert(sys.argv[1], sys.argv[2])
    print(f"wrote {n} ACP events -> {sys.argv[2]} (+ .meta.json) | session {meta.get('id')} "
          f"model={meta.get('model')} end_reason={meta.get('end_reason')}")
