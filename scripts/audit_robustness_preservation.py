"""Read-only audit of old artifacts/environments; save evidence outside frozen run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traceguard.engineering import sha

root = Path('artifacts/trace-robustness')
destination = root / 'preservation-check.json'
if destination.exists():
    raise ValueError('Preserve the existing audit; choose a new evidence file for another audit.')
baseline = json.loads((root / 'preservation-before.json').read_text(encoding='utf-8'))
allowed = {'traceguard/features.py', 'traceguard/behavior.py', 'traceguard/records.py',
           'traceguard/detector.py', 'traceguard/__main__.py', 'traceguard/server.py',
           'traceguard/monitor.html', 'traceguard/monitor.js', 'README.md',
           'docs/DEVELOPMENT_STATUS.md', 'docs/thesis/PROJECT_CONTEXT.md'}
changed = [name for name, digest in baseline.items() if not Path(name).is_file() or sha(name) != digest]
unexpected = [name for name in changed if Path(name).as_posix() not in allowed]
clean = json.loads(Path('artifacts/lora-trajectory/clean-checkpoint-before.json').read_text(encoding='utf-8'))
environments = {}
for environment, name in [('.venv', 'venv'), ('.venv-gpu', 'venv-gpu'), ('.venv-train', 'venv-train')]:
    before = Path(f'artifacts/trace-engineering/{name}-packages.txt').read_text(encoding='utf-8-sig')
    proc = subprocess.run([str(Path(environment) / 'Scripts/python.exe'), '-m', 'pip', 'freeze'],
                          capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if proc.returncode:
        raise RuntimeError(f'Could not inspect {environment}: pip freeze exit {proc.returncode}')
    normalized = lambda text: '\n'.join(sorted(s.strip() for s in text.splitlines() if s.strip()))
    before, after = normalized(before), normalized(proc.stdout.decode('utf-8'))
    environments[environment] = {'matches_pre_5B_package_pins': before == after,
                                'package_count': len(after.splitlines()),
                                'normalized_pins_sha256': hashlib.sha256(after.encode()).hexdigest()}
run = root / 'run-001'
bundle = json.loads((run / 'bundle.json').read_text(encoding='utf-8'))
result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
          'old_files_checked': len(baseline), 'old_files_unchanged': len(baseline)-len(changed),
          'intentional_runtime_and_current_document_changes': changed,
          'unexpected_old_file_changes': unexpected,
          'original_clean_weights_unchanged': sha(clean['path']) == clean['sha256'],
          'all_frozen_bundle_hashes_match': all(sha(run / name) == digest for name, digest in bundle['files_sha256'].items()),
          'environments': environments}
result['passed'] = not unexpected and result['original_clean_weights_unchanged'] and result['all_frozen_bundle_hashes_match'] and all(v['matches_pre_5B_package_pins'] for v in environments.values())
with destination.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['passed'] else 1)
