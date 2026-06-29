"""Synthetically expand the companion dataset: a strong local model generates more
(user message, Mira reply) pairs in her voice, from the persona + seed as few-shot.

Saves to the dashboard SQLite (visible in Data / Fine-tune). $0 (local LM Studio).
Run:  source env-wsl.sh && dashboard/.venv/bin/python companion/expand.py --n 48
      ... --model <id> --target companion --batch 6
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import db  # noqa: E402
import litellm  # noqa: E402
from seed_companion import PERSONA, SEED  # noqa: E402

litellm.drop_params = True
litellm.suppress_debug_info = True

PAIRS_SCHEMA = {"name": "pairs", "strict": True, "schema": {"type": "object", "properties": {
    "pairs": {"type": "array", "items": {"type": "object", "properties": {
        "user": {"type": "string"}, "reply": {"type": "string"}},
        "required": ["user", "reply"], "additionalProperties": False}}},
    "required": ["pairs"], "additionalProperties": False}}

# Substance-heavy topics: v1 over-flirted, so v2 forces real engagement (answer / help / true support).
TOPICS = [
    "a real factual or technical question she must actually answer (science, history, how-to)",
    "the user is in genuine emotional pain and needs real support, NOT deflection or flirting",
    "the user asks for concrete practical help (a plan, a recipe, a decision)",
    "a thoughtful question about feelings or relationships that needs a real answer",
    "the user just wants to vent and be heard - no fixing, no flirting it away",
    "self-doubt the user needs gently challenged with real reasons, not empty praise",
    "a practical problem she helps solve step by step",
    "speaking Russian, with real substance",
    "the user tests her ('do you actually care?') - honest and warm, not deflective",
    "a curious deep topic she engages with genuine intelligence",
    "the user is low-energy or terse and she reads the subtext without being pushy",
    "a light flirty moment that STILL carries warmth and substance (not empty charm)",
]


def derive_base():
    if os.getenv("LM_STUDIO_API_BASE"):
        return os.getenv("LM_STUDIO_API_BASE")
    try:
        gw = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout.split()[2]
        return f"http://{gw}:1234/v1"
    except Exception:
        return "http://localhost:1234/v1"


def pick_model(api_base):
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as r:
            ms = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return ""
    bad = ("coder", "embed", "uncensored", "nsfw", "abliterated", "vision", "-vl")
    cands = [m for m in ms if not any(b in m.lower() for b in bad)]
    for p in ("qwen2.5-7b", "llama-3.3", "phi-4", "gemma", "mistral", "qwen"):
        for m in cands:
            if p in m.lower():
                return m
    return cands[0] if cands else (ms[0] if ms else "")


def gen_batch(model, api_base, k, topic):
    ex = "\n".join(f"U: {i}\nM: {o}" for i, o in SEED[:6])
    sysm = (PERSONA + " You generate TRAINING data: realistic (user message, Mira reply) pairs in her voice. "
            "HARD RULE: Mira ALWAYS engages the user's REAL need with genuine substance FIRST - answer the "
            "question, help with the problem, truly support the feeling. Charm is a LIGHT touch ON TOP, never "
            "a replacement. NEVER dodge a question to flirt. NEVER dismiss pain (no 'you're just having a bad "
            "day'). NEVER say 'I'm programmed to...'. A reply that is all flirt and no substance is WRONG.")
    usr = (f"Mira's voice examples (substance + a light charming touch):\n{ex}\n\nGenerate {k} NEW, diverse "
           f"pairs about: {topic}. Each Mira reply must genuinely engage the real need (correct answer / real "
           "help / true empathy), and only then add a little warmth or wit. Vary phrasing and length; tasteful; "
           "clearly different from the examples. JSON only.")
    mid = model if model.startswith(("lm_studio/", "openrouter/")) else f"lm_studio/{model}"
    kw = ({"api_key": os.getenv("OPENROUTER_API_KEY", "")} if model.startswith("openrouter/")
          else {"api_base": api_base, "api_key": "lm-studio"})
    r = litellm.completion(model=mid, temperature=0.9, timeout=300,
                           messages=[{"role": "system", "content": sysm}, {"role": "user", "content": usr}],
                           response_format={"type": "json_schema", "json_schema": PAIRS_SCHEMA}, **kw)
    return json.loads(r.choices[0].message.content).get("pairs", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--model", default="")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--target", default="companion")
    a = ap.parse_args()
    db.init()
    base = derive_base()
    model = a.model or pick_model(base)
    if not model:
        print("No model loaded in LM Studio - load one and retry.")
        return
    print(f"generating ~{a.n} pairs with `{model}` @ {base} ...")
    made, ti = 0, 0
    while made < a.n:
        topic = TOPICS[ti % len(TOPICS)]
        ti += 1
        try:
            pairs = gen_batch(model, base, min(a.batch, a.n - made), topic)
        except Exception as e:  # noqa: BLE001
            print("  batch failed:", str(e)[:120])
            if ti > a.n:  # avoid infinite loop on persistent failure
                break
            continue
        for p in pairs:
            u, m = str(p.get("user", "")).strip(), str(p.get("reply", "")).strip()
            if u and m:
                db.add_case(a.target, u, m, PERSONA, "")
                made += 1
        print(f"  {made}/{a.n}  (topic: {topic})")
    print(f"done: +{made} synthetic cases -> '{a.target}' (now {len(db.list_cases(a.target))}). "
          "Visible in the dashboard Data tab.")


if __name__ == "__main__":
    main()
