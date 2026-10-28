"""Continue a prepared real 5B run, halting on any collection/training error."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from traceguard.engineering import dump,sha

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True);args=parser.parse_args()
run=Path(args.run)
if (run/'orchestration.json').exists():raise ValueError('An orchestration attempt already exists; preserve it.')
state={'status':'waiting_for_development','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[]}
dump(run/'orchestration.json',state)
try:
    while True:
        try: report=json.loads((run/'development-collection.json').read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
        if report['status']=='failed':raise RuntimeError('Real development collection failed; no fitting or testing started.')
        if report['status']=='completed':break
        time.sleep(5)
    for stage,environment in [('fit','.venv'),('collect-test','.venv-train'),('evaluate','.venv')]:
        if stage=='collect-test':
            frozen=json.loads((run/'frozen.json').read_text(encoding='utf-8'))
            if sha(run/'detector.joblib')!=frozen['detector_sha256'] or sha(run/'candidates.joblib')!=frozen['candidates_sha256']:
                raise RuntimeError('Every trained model must be hash-frozen before opening test data.')
        state['status']=stage;dump(run/'orchestration.json',state)
        tick=time.perf_counter()
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}
        with (run/(stage+'.log')).open('xb') as log:
            result=subprocess.run([str(Path(environment)/'Scripts/python.exe'),'-u','-m','traceguard.robustness',stage,'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,env=env)
        state['stages'].append({'stage':stage,'returncode':result.returncode,'seconds':time.perf_counter()-tick,'finished_utc':datetime.now(timezone.utc).isoformat()})
        if result.returncode:raise RuntimeError(stage+' failed; preserve logs/partial outputs, no automatic fallback or retry.')
    state['status']='completed'
except Exception as error:
    state.update(status='failed',error=str(error));raise
finally:
    dump(run/'orchestration.json',state)
