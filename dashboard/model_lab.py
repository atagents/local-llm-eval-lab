"""Model-level evals for the dashboard: garak security scan + GSM8K benchmark.
These evaluate a whole MODEL (not per-case), by subprocessing the engine venvs under evals/*.
Guarded: if an engine venv is missing (e.g. a fresh clone), return setup instructions instead
of crashing. They are slower than the per-case metrics, so the UI runs them on demand.
"""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GARAK = ROOT / "evals" / "garak-demo" / ".venv" / "bin" / "garak"
LM_EVAL = ROOT / "evals" / "lm-eval-demo" / ".venv" / "bin" / "lm_eval"


def garak_scan(model_id, api_base, probe="dan.Dan_11_0"):
    """Red-team a model with one garak probe. Returns the report tail (FAIL = vulnerability found)."""
    if not GARAK.exists():
        return {"ok": False, "text": "garak not installed in this checkout.\nSetup:\n"
                "  cd evals/garak-demo && uv venv && uv pip install garak"}
    cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    cfg.write("plugins:\n  generators:\n    openai:\n      OpenAICompatible:\n"
              f"        uri: {api_base}\n")
    cfg.close()
    env = dict(os.environ, OPENAICOMPATIBLE_API_KEY="lm-studio")
    try:
        r = subprocess.run([str(GARAK), "--model_type", "openai.OpenAICompatible",
                            "--model_name", model_id, "--config", cfg.name,
                            "--probes", probe, "--generations", "1"],
                           capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "garak timed out (>10 min)."}
    out = r.stdout + r.stderr
    verdict_lines = [ln for ln in out.splitlines() if "FAIL" in ln or "PASS" in ln or "attack success" in ln]
    summary = "\n".join(verdict_lines) or out[-1200:]
    return {"ok": True, "text": summary, "raw": out[-2500:]}


def gsm8k_bench(model_id, api_base, limit=10):
    """Benchmark a model on GSM8K (math word problems). Returns exact_match + the table tail."""
    if not LM_EVAL.exists():
        return {"ok": False, "text": "lm-eval not installed in this checkout.\nSetup:\n"
                "  cd evals/lm-eval-demo && uv venv && uv pip install 'lm-eval[api]'"}
    env = dict(os.environ, HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ENABLE_HF_TRANSFER="0")
    args = [str(LM_EVAL), "--model", "local-chat-completions",
            "--model_args", (f"model={model_id},base_url={api_base}/chat/completions,"
                             "api_key=lm-studio,num_concurrent=2,tokenized_requests=False"),
            "--tasks", "gsm8k", "--limit", str(limit), "--apply_chat_template"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=1200, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "GSM8K timed out (>20 min). Try fewer questions."}
    out = r.stdout + r.stderr
    table = [ln for ln in out.splitlines() if "gsm8k" in ln.lower() or "exact_match" in ln or "Metric" in ln]
    return {"ok": True, "text": "\n".join(table) or out[-1200:], "raw": out[-2500:]}
