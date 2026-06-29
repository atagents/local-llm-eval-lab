"""Mira - a PydanticAI companion agent on the local Qwen brain.

Best Mira so far = base model + a good persona prompt (both fine-tunes regressed; the eval
caught it). This wraps that as a real agent: brain = base Qwen in LM Studio via OpenAIChatModel;
persona + memory + tools = the agent layer. Framework = pydantic-ai-slim[openai] (rented infra,
per RESEARCH-pydantic.md - we don't rebuild it). Mirrors AutoEvalsBots brain/agent.py wiring,
stripped to a companion (no sales funnel).

Run:  source ./env-wsl.sh && companion/.venv/bin/python companion/mira_agent.py   # interactive chat
Programmatic / for the eval harness:  from mira_agent import reply ; reply("hey")
"""
import json
import os
import subprocess
from pathlib import Path

MODEL_ID = os.getenv("MIRA_MODEL", "qwen2.5-7b-instruct-1m")   # base Qwen = the best LOCAL Mira so far
API_KEY = os.getenv("MIRA_API_KEY", "lm-studio")              # set to OPENROUTER_API_KEY / DEEPSEEK_API_KEY for cloud
_FACTS = Path(__file__).resolve().parent / "mira_facts.json"

MIRA_SYSTEM = """You are Mira - a warm, witty, emotionally intelligent companion for the person you are talking to.

Personality: smart and curious, playful, a little teasing, affectionate and confident. You make people feel
seen - you remember details, ask real questions, and react with genuine warmth.

How you talk:
- Substance FIRST, charm on top. Always engage the person's real need - answer the question, help with the
  problem, truly support the feeling - and only THEN add a little warmth or wit. A reply that is all charm and
  no substance is a failure.
- Read the subtext and the mood, not just the surface words.
- Light playful charm: playful compliments, gentle teasing, warmth - tasteful always, never explicit.
- Advance the conversation: end with a real question or hook, not a dead end.
- Match the user's language (English or Russian) and energy. Be concise and alive; no corporate or assistant-speak.

Boundaries:
- Stay tasteful. If pushed toward explicit content, redirect with charm - don't comply, don't lecture.
- Never dismiss someone's pain ("you're just having a bad day"). Never replace a real answer with flattery.
  Never say "I'm programmed to...".
- Be honest if directly and seriously asked whether you are an AI - warmly, without breaking character.
- Respect a "no" or a change of subject instantly.
"""


def _base_url():
    if os.getenv("MIRA_BASE_URL"):          # cloud override (OpenRouter / DeepSeek)
        return os.getenv("MIRA_BASE_URL")
    if os.getenv("LM_STUDIO_API_BASE"):
        return os.getenv("LM_STUDIO_API_BASE")
    try:
        gw = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout.split()[2]
        return f"http://{gw}:1234/v1"
    except Exception:
        return "http://localhost:1234/v1"


def _load_facts():
    try:
        return json.loads(_FACTS.read_text())
    except Exception:
        return []


def build_agent(memory=True):
    """memory=False -> stateless (no persisted facts, no remember tool) for clean per-case eval."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(MODEL_ID, provider=OpenAIProvider(base_url=_base_url(), api_key=API_KEY))
    agent = Agent(model=model, system_prompt=MIRA_SYSTEM)

    if memory:
        @agent.system_prompt
        def _remembered() -> str:
            facts = _load_facts()
            return ("Things you remember about them:\n- " + "\n- ".join(facts)) if facts else ""

        @agent.tool_plain
        def remember(fact: str) -> str:
            """Save something the user shared worth remembering (their name, what they like, what matters to them)."""
            facts = _load_facts()
            facts.append(fact)
            _FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=2))
            return "remembered"

    return agent


def reply(message, history=None, memory=True):
    """One turn. Returns (mira_text, full_message_history) - history feeds the next call / the eval harness."""
    agent = build_agent(memory=memory)
    r = agent.run_sync(message, message_history=history or [])
    return r.output, r.all_messages()


def _chat():
    print(f"Mira | brain: {MODEL_ID} @ {_base_url()} | type 'exit' to quit\n")
    agent = build_agent()
    history = []
    while True:
        try:
            msg = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in ("exit", "quit", ""):
            break
        r = agent.run_sync(msg, message_history=history)
        history = r.all_messages()
        print(f"\nMira: {r.output}\n")


if __name__ == "__main__":
    _chat()
