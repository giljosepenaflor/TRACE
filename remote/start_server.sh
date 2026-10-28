#!/usr/bin/env bash
set -euo pipefail

# Run from a private Vast terminal. Secrets are read without echoing and never
# placed in command-line arguments or saved alongside the model code.
if [[ -z "${HF_TOKEN:-}" && -f /root/.config/trace/hf_token ]]; then
  HF_TOKEN="$(cat /root/.config/trace/hf_token)"
fi
if [[ -z "${TRACE_API_TOKEN:-}" && -f /root/.config/trace/api_token ]]; then
  TRACE_API_TOKEN="$(cat /root/.config/trace/api_token)"
fi
# PowerShell's text pipeline writes CRLF. Bash command substitution removes the
# LF but leaves one trailing CR; strip that transport character only.
HF_TOKEN="${HF_TOKEN%$'\r'}"
TRACE_API_TOKEN="${TRACE_API_TOKEN%$'\r'}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  read -r -s -p 'Hugging Face read token: ' HF_TOKEN
  printf '\n'
fi
if [[ -z "${TRACE_API_TOKEN:-}" ]]; then
  read -r -s -p 'TRACE API token (same value used locally): ' TRACE_API_TOKEN
  printf '\n'
fi
export HF_TOKEN TRACE_API_TOKEN
cd /workspace/trace
exec /workspace/trace-env/bin/python trace_inference_server.py
