#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/trace-portable
if [[ ! -x "$ROOT/env/bin/python" ]]; then
  echo 'Run portable_setup.sh before starting TRACE.' >&2
  exit 1
fi
if [[ ! -s /root/.config/trace/api_token ]]; then
  echo 'Missing private TRACE API credential.' >&2
  exit 1
fi
TRACE_API_TOKEN="$(cat /root/.config/trace/api_token)"
TRACE_API_TOKEN="${TRACE_API_TOKEN%$'\r'}"
export TRACE_API_TOKEN
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRACE_BASE_SNAPSHOT="$ROOT/hf-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
export TRACE_ADAPTER_SNAPSHOT="$ROOT/hf-cache/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_MTBA/snapshots/8652ad60f1605605642c99fd680a15ecccf1b522"
cd "$ROOT/app"
exec "$ROOT/env/bin/python" trace_inference_server.py
