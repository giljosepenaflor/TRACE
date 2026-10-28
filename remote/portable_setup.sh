#!/usr/bin/env bash
# Prepare a copied TRACE bundle without loading the 7B model.
set -euo pipefail

ROOT=/workspace/trace-portable
if [[ "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" != "$ROOT" ]]; then
  echo 'Copy the complete bundle to /workspace/trace-portable before setup.' >&2
  exit 1
fi

if [[ ! -x /venv/main/bin/python ]]; then
  echo 'This bundle requires the Vast PyTorch template with /venv/main/bin/python (Python 3.12). Use the same template as the source instance.' >&2
  exit 1
fi
if [[ ! -e "$ROOT/env/bin/python" ]]; then
  ln -sfn /venv/main/bin/python "$ROOT/env/bin/python"
fi
if [[ ! -x "$ROOT/env/bin/python" ]]; then
  echo 'The copied isolated Python environment is unusable.' >&2
  exit 1
fi

/venv/main/bin/python "$ROOT/app/portable_backend.py" verify "$ROOT"
"$ROOT/env/bin/python" - "$ROOT" <<'PY'
import importlib.metadata as md
import sys
from pathlib import Path

root = Path(sys.argv[1])
if Path(sys.prefix).resolve() != (root / 'env').resolve():
    raise RuntimeError('Copied Python interpreter is not using the isolated TRACE environment')
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('The copied environment requires Python 3.12')
expected = {
    'numpy': '1.26.4', 'transformers': '4.46.3', 'tokenizers': '0.20.3',
    'peft': '0.12.0', 'accelerate': '0.34.0', 'bitsandbytes': '0.50.2',
    'huggingface-hub': '0.36.2', 'sentencepiece': '0.2.1',
}
for package, version in expected.items():
    actual = md.version(package)
    if actual != version:
        raise RuntimeError(f'{package} must be {version}, found {actual}')
import torch
import transformers, peft, accelerate, bitsandbytes, sentencepiece  # noqa: F401
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable on this instance')
value = (torch.tensor([2.0], device='cuda') + 3.0).item()
if value != 5.0:
    raise RuntimeError('CUDA tensor operation failed')
layer = bitsandbytes.nn.Linear4bit(
    64, 64, bias=False, compute_dtype=torch.float16, quant_type='nf4').to('cuda')
output = layer(torch.ones((1, 64), device='cuda', dtype=torch.float16))
torch.cuda.synchronize()
if not torch.isfinite(output).all().item():
    raise RuntimeError('4-bit NF4 CUDA operation failed')
print('Environment verified:', sys.version.split()[0],
      'torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))
PY
"$ROOT/env/bin/python" -m pip check

if [[ "${1:-}" == --check-only ]]; then
  echo 'Bundle and environment verified; no service or credentials changed.'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: portable_setup.sh [--check-only]' >&2
  exit 2
fi
path=/root/.config/trace/api_token
if [[ ! -s "$path" ]] || [[ "$(stat -c %a "$path")" != 600 ]]; then
  echo 'Missing root-only API token; run setup-vast-portable.ps1 from Windows.' >&2
  exit 1
fi
if ! command -v supervisorctl >/dev/null; then
  echo 'This Vast template needs supervisorctl for the managed TRACE service.' >&2
  exit 1
fi

cat > /etc/supervisor/conf.d/trace_llama2.conf <<'CONF'
[program:trace_llama2]
command=/bin/bash /workspace/trace-portable/app/start_server.sh
directory=/workspace/trace-portable/app
autostart=false
autorestart=unexpected
startsecs=10
stopasgroup=true
killasgroup=true
environment=HF_HOME="/workspace/trace-portable/hf-cache",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF
supervisorctl reread
supervisorctl update
echo 'TRACE service registered. Setup itself did not load the GPU model; use the startup command when ready.'
