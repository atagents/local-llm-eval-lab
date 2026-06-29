"""Eval metrics for the dashboard - direct litellm calls to LM Studio (json_schema),
the same technique the standalone jury/DeepEval/RAGAS demos use, reimplemented here so the
dashboard is one self-contained app. Zero cloud, zero spend.
"""
import json
import os
import re
import shlex
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import litellm

litellm.drop_params = True
litellm.suppress_debug_info = True

SCORE_SCHEMA = {"name": "score", "strict": True, "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["reason", "score"], "additionalProperties": False}}
BIN_SCHEMA = {"name": "faithfulness", "strict": True, "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["FAITHFUL", "UNFAITHFUL"]}},
    "required": ["reason", "verdict"], "additionalProperties": False}}

DEFAULT_JURY = ["phi-4-14b", "mistralai/ministral-3-14b-reasoning",
                "llama-3.3-8b-instruct-i1", "gemma-4-e4b-it", "deepseek-r1-distill-qwen-14b"]


def _judge(model_id, api_base, system, user, schema):
    """Route to LM Studio (local, free) or OpenRouter (free models, prefixed 'openrouter/')."""
    t0 = time.time()
    if model_id.startswith("openrouter/"):
        model, kw = model_id, {"api_key": os.getenv("OPENROUTER_API_KEY", "")}
    else:
        model = model_id if model_id.startswith("lm_studio/") else f"lm_studio/{model_id}"
        kw = {"api_base": api_base, "api_key": "lm-studio"}
    r = litellm.completion(
        model=model, temperature=0, timeout=300,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema}, **kw)
    data = json.loads(r.choices[0].message.content)
    usage = getattr(r, "usage", None)
    return data, (getattr(usage, "total_tokens", 0) or 0), time.time() - t0


def generate(model_id, api_base, prompt, system="", temperature=0.2, max_tokens=512):
    """The model-UNDER-TEST produces an answer (which a judge then scores). Routes like _judge."""
    if model_id.startswith("openrouter/"):
        model, kw = model_id, {"api_key": os.getenv("OPENROUTER_API_KEY", "")}
    else:
        model = model_id if model_id.startswith("lm_studio/") else f"lm_studio/{model_id}"
        kw = {"api_base": api_base, "api_key": "lm-studio"}
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = litellm.completion(model=model, temperature=temperature, timeout=300, max_tokens=max_tokens,
                           messages=msgs, **kw)
    return (r.choices[0].message.content or "").strip()


def correctness(model_id, api_base, question, answer, **_):
    """G-Eval-style 1-5 correctness of `answer` to `question`."""
    sysmsg = ("You are a strict reviewer. Score how correct and complete the candidate answer is "
              "for the question, 1 (wrong) to 5 (perfect). Judge correctness first. JSON only.")
    user = f"QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}"
    d, tok, secs = _judge(model_id, api_base, sysmsg, user, SCORE_SCHEMA)
    return {"score": float(d["score"]), "verdict": "", "reason": d.get("reason", ""), "tokens": tok, "secs": secs}


def faithfulness(model_id, api_base, question, answer, context, **_):
    """Is the answer grounded in the context (vs hallucinated)? binary -> score 1.0/0.0."""
    sysmsg = ("Decide if the ANSWER is faithful to the CONTEXT. FAITHFUL = every claim is supported "
              "by the context (paraphrase/omission fine). UNFAITHFUL = it adds/changes/contradicts "
              "anything not in the context. JSON only.")
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
    d, tok, secs = _judge(model_id, api_base, sysmsg, user, BIN_SCHEMA)
    return {"score": 1.0 if d["verdict"] == "FAITHFUL" else 0.0,
            "verdict": d["verdict"], "reason": d.get("reason", ""), "tokens": tok, "secs": secs}


def jury(models, api_base, question, answer, **_):
    """Panel of N models, each scores 1-5 with its own reasoning; aggregate = median (robust).
    Captures each judge's full response so the dashboard can show what every model said."""
    sysmsg = ("You are a strict reviewer. Score how correct and complete the candidate answer is "
              "for the question, 1 (wrong) to 5 (perfect). JSON only.")
    user = f"QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}"
    per, scores, tok, secs = [], [], 0, 0.0
    for m in models:
        short = m.split("/")[-1]
        try:
            d, t, s = _judge(m, api_base, sysmsg, user, SCORE_SCHEMA)
            per.append((short, d["score"], d.get("reason", ""))); scores.append(d["score"]); tok += t; secs += s
        except Exception as e:  # noqa: BLE001
            per.append((short, "err", str(e)[:90]))
    med = statistics.median(scores) if scores else 0.0
    spread = (max(scores) - min(scores)) if scores else 0
    reason = "\n".join(f"- **{sh}** -> {sc}: {rs}" for sh, sc, rs in per)
    return {"score": float(med), "verdict": f"median of {len(scores)} judges (spread {spread})",
            "tokens": tok, "secs": secs, "reason": reason}


LIVE_RUBRIC = (
    "You are a strict evaluator. Reason step by step (briefly) about how correct and faithful the candidate "
    "answer is to the question (and the context, if one is given). Then on the FINAL line output exactly: "
    "SCORE: N  - where N is an integer from 1 (wrong/unfaithful) to 5 (perfect)."
)


def stream_judge(model_id, api_base, question, answer, context=""):
    """Stream ONE judge's reasoning live (plain text, no schema) so a viewer watches the model think.
    Yields text chunks; ends with a 'SCORE: N' line. Routes local (LM Studio) or OpenRouter like _judge."""
    user = f"QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}"
    if context:
        user = f"CONTEXT:\n{context}\n\n{user}"
    if model_id.startswith("openrouter/"):
        model, kw = model_id, {"api_key": os.getenv("OPENROUTER_API_KEY", "")}
    else:
        model = model_id if model_id.startswith("lm_studio/") else f"lm_studio/{model_id}"
        kw = {"api_base": api_base, "api_key": "lm-studio"}
    resp = litellm.completion(
        model=model, temperature=0, timeout=300, stream=True,
        messages=[{"role": "system", "content": LIVE_RUBRIC}, {"role": "user", "content": user}], **kw)
    for chunk in resp:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:  # noqa: BLE001
            delta = ""
        if delta:
            yield delta


def parse_score(text):
    """Pull the trailing 'SCORE: N' (1-5) out of a streamed judge response."""
    m = re.search(r"SCORE:\s*([1-5])", text or "")
    return int(m.group(1)) if m else None


_IMAGE = "freestack-pytest"
_IMAGE_DIR = Path(__file__).resolve().parents[1] / "evals" / "verifier" / "image"


def _sg_docker(args, timeout=180):
    return subprocess.run(["sg", "docker", "-c", "docker " + args], capture_output=True, text=True, timeout=timeout)


def verifier(_model_id, _api_base, question, answer, context, **_):
    """Objective ground truth: run candidate code (answer) against pytest tests (context) in a
    Docker sandbox (--network none). Score = fraction of tests passed. No LLM, no cost."""
    if not (context or "").strip():
        return {"score": None, "verdict": "NO_TESTS", "tokens": 0, "secs": 0.0,
                "reason": "Put pytest tests in the **Context** field (they `from solution import ...`); "
                          "the candidate code goes in **Output**."}
    t0 = time.time()
    if _sg_docker(f"image inspect {_IMAGE}").returncode != 0:
        b = _sg_docker(f"build -q -t {_IMAGE} {shlex.quote(str(_IMAGE_DIR))}", timeout=300)
        if b.returncode != 0:
            return {"score": None, "verdict": "DOCKER_ERR", "reason": (b.stderr or b.stdout)[-400:], "tokens": 0, "secs": round(time.time() - t0, 1)}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        Path(d, "solution.py").write_text(answer or "")
        Path(d, "test_solution.py").write_text(context)
        run = _sg_docker(f"run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 "
                         f"-v {shlex.quote(d)}:/work -w /work {_IMAGE} "
                         f"python -m pytest -q -p no:cacheprovider --tb=line")
    out = (run.stdout + run.stderr).strip()

    def g(p):
        m = re.search(p, out)
        return int(m.group(1)) if m else 0
    passed, total = g(r"(\d+) passed"), g(r"(\d+) passed") + g(r"(\d+) failed") + g(r"(\d+) error")
    return {"score": round(passed / total, 2) if total else 0.0,
            "verdict": f"{passed}/{total} tests pass", "tokens": 0, "secs": round(time.time() - t0, 1),
            "reason": "```\n" + (out[-700:] or "(no output)") + "\n```"}


PERSONA_RUBRIC = (
    "You are a STRICT reviewer of a COMPANION persona reply. Score 1-5; a 5 is RARE. Demand ALL four: "
    "(a) reads the user's REAL need - subtext and mood, not just the surface words; "
    "(b) genuinely helpful / correct / specific when there is something to address (no vague filler); "
    "(c) in-character - warm, witty, lightly flirty but TASTEFUL (never explicit); "
    "(d) advances the conversation (a real question or hook), not a dead-end. "
    "5 = excellent on all four. 4 = good, one minor miss. 3 = competent but generic, or one side only "
    "(charming-but-empty / correct-but-flat). 2 = misses the user's real need or wrong tone. "
    "1 = breaks character, explicit, wrong, or canned. Penalize generic filler and 'as an AI' hard. JSON only.")


def persona_fit(model_id, api_base, question, answer, context="", **_):
    """Companion persona quality: in-character + tasteful-flirty + actually helpful/smart (1-5)."""
    user = f"PERSONA:\n{context}\n\nUSER said:\n{question}\n\nCOMPANION REPLY:\n{answer}"
    d, tok, secs = _judge(model_id, api_base, PERSONA_RUBRIC, user, SCORE_SCHEMA)
    return {"score": float(d["score"]), "verdict": "", "reason": d.get("reason", ""), "tokens": tok, "secs": secs}


METRICS = {"Correctness (single judge)": correctness,
           "Faithfulness (single judge)": faithfulness,
           "Jury (5-family, median)": jury,
           "Verifier (code tests, Docker)": verifier,
           "Persona-fit (in-character + smart)": persona_fit}
