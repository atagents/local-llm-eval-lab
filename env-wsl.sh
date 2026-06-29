# env-wsl.sh - source this before running the harness on the WSL host.
#   . ./env-wsl.sh
# Verified working 2026-06-23 (smoke passed end-to-end on qwen2.5-coder-7b).
#
# Topology: Claude Code + mini-swe-agent in WSL2 -> models in LM Studio on the
# Windows host (OpenAI endpoint :1234). WSL is NAT mode (not mirrored), so
# `localhost` does NOT reach the host - we derive the host IP from the default route.

# Host IP = WSL NAT default gateway (the Windows-side vEthernet adapter).
# Derived dynamically because the NAT subnet can change across `wsl --shutdown`.
HOST_IP="$(ip route show default | awk '{print $3}')"   # was <lm-studio-host> on 2026-06-23

export PYTHONUTF8=1                                   # harmless on Linux; matters on Windows
export LM_STUDIO_API_BASE="http://${HOST_IP}:1234/v1"
# Auth is currently OPEN on this LM Studio server (chat/completions returns 200
# with no Authorization header), so any non-empty placeholder satisfies litellm.
# If the permission token gets enforced later, set the real token here instead
# (secret - do not commit / do not paste in chat):  export LM_STUDIO_API_KEY=<token>
export LM_STUDIO_API_KEY="${LM_STUDIO_API_KEY:-lm-studio}"
export MSWEA_COST_TRACKING=ignore_errors             # local model has no price in litellm
export MSWEA_MODEL_NAME="lm_studio/qwen2.5-coder-7b-instruct"   # agent model

# Convenience: path to the venv python that has minisweagent installed.
export MSWEA_PY="$HOME/projects/my-unsloth-finetune/mini-swe-agent/.venv/bin/python"

echo "LM Studio @ ${LM_STUDIO_API_BASE}  | agent model: ${MSWEA_MODEL_NAME}"
