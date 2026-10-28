"""Verify prior artifacts and environments without changing frozen experiments."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from traceguard.engineering import sha

root=Path('artifacts/trace-live')
before=json.loads((root/'preservation-before.json').read_text(encoding='utf-8'))
allowed={'traceguard/server.py','traceguard/__main__.py','README.md','docs/DEVELOPMENT_STATUS.md','docs/thesis/PROJECT_CONTEXT.md'}
changed=[p for p,d in before.items() if not Path(p).is_file() or sha(p)!=d]
unexpected=[p for p in changed if Path(p).as_posix() not in allowed]
environments={}
normalize=lambda s:'\n'.join(sorted(v.strip() for v in s.splitlines() if v.strip()))
for directory,baseline in [('.venv','venv'),('.venv-gpu','venv-gpu'),('.venv-train','venv-train')]:
    expected=normalize(Path(f'artifacts/trace-engineering/{baseline}-packages.txt').read_text(encoding='utf-8-sig'))
    result=subprocess.run([str(Path(directory)/'Scripts/python.exe'),'-m','pip','freeze'],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if result.returncode:raise RuntimeError('Cannot inspect '+directory)
    actual=normalize(result.stdout.decode('utf-8'))
    environments[directory]={'unchanged':expected==actual,'package_count':len(actual.splitlines()),
                             'pins_sha256':hashlib.sha256(actual.encode()).hexdigest()}
clean=json.loads(Path('artifacts/lora-trajectory/clean-checkpoint-before.json').read_text())
bundles={}
for directory in ('artifacts/trace-engineering/run-001','artifacts/trace-robustness/run-001'):
    p=Path(directory);manifest=json.loads((p/'bundle.json').read_text())
    bundles[directory]=all(sha(p/name)==digest for name,digest in manifest['files_sha256'].items())
evidence={'checked_utc':datetime.now(timezone.utc).isoformat(),'old_files_checked':len(before),
          'old_files_unchanged':len(before)-len(changed),'intentional_code_and_current_doc_changes':changed,
          'unexpected_old_file_changes':unexpected,'frozen_bundles_unchanged':bundles,
          'original_clean_weights_unchanged':sha(clean['path'])==clean['sha256'],'environments':environments}
evidence['passed']=not unexpected and all(bundles.values()) and evidence['original_clean_weights_unchanged'] and all(v['unchanged'] for v in environments.values())
with (root/'preservation-final.json').open('x',encoding='utf-8') as stream:json.dump(evidence,stream,indent=2)
print(json.dumps(evidence,indent=2))
raise SystemExit(0 if evidence['passed'] else 1)
