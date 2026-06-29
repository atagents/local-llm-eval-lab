"""Run mini-swe-agent on one task against a LOCAL model (LM Studio) at zero cost.

Why a custom runner instead of `minisweagent.run.hello_world`:
- hello_world hardcodes `LitellmModel`, which forces native tool-calls. Local GGUF
  models (LM Studio) usually don't emit tool_calls -> RepeatedFormatError.
  We use `LitellmTextbasedModel`, which parses a ```mswea_bash_command``` block from text.

Required env (set before running):
  PYTHONUTF8=1                 # Windows console: avoid cp1252 crash on emoji banner
  LM_STUDIO_API_BASE=http://<host>:1234/v1
  LM_STUDIO_API_KEY=<token>    # the LM Studio permission token
  MSWEA_COST_TRACKING=ignore_errors   # local models have no price in litellm

Usage:
  python local_smoke.py "your task here"   (model from MSWEA_MODEL_NAME or default below)
"""

import json
import logging
import sys

import yaml

logging.basicConfig(level=logging.WARNING)
for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import os

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

MODEL = os.getenv("MSWEA_MODEL_NAME", "lm_studio/qwen2.5-coder-7b-instruct")
TASK = sys.argv[1] if len(sys.argv) > 1 else (
    "Create a file named hello.txt in the current directory containing exactly "
    "the text: HELLO WORLD. Then verify it exists by listing the directory. Then finish."
)

cfg = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())["agent"]
cfg["step_limit"] = 12  # safety cap so a non-finishing local model can't loop forever
agent = DefaultAgent(LitellmTextbasedModel(model_name=MODEL), LocalEnvironment(), **cfg)

result = None
try:
    result = agent.run(TASK)
except Exception as e:  # noqa: BLE001 - smoke run, surface whatever happens
    print("RUN RAISED:", type(e).__name__, "-", str(e)[:300])

print("RUN RESULT:", repr(result)[:300])
for i, m in enumerate(agent.messages):
    content = m.get("content", "")
    if isinstance(content, list):
        content = json.dumps(content)
    print(f"\n----- [{i}] {m.get('role', '?')} -----")
    print(str(content)[:1800])
