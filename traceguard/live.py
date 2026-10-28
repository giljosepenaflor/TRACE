"""Live orchestration around the frozen F0 feature path; no labels or fitting."""
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from uuid import uuid4

import numpy as np

from .engineering import sha
from .features import FEATURE_NAMES, generation_features
from .live_worker import check_prompt
from .robustness_protocol import checkpoint_identity, GENERATION
from .runtime import resource_snapshot


def audit_development(artifact, run):
    """Numerical compatibility only, using eight model-selection records; no tuning."""
    from .engineering import read_json
    from .records import read_records
    from .robustness_detector import matrices, score_matrices
    run=Path(run)
    rows=[r for r in read_records(run/'development.jsonl') if r['robustness_partition']=='model_selection'][:8]
    saved={r['sample_id']:r['scores']['F0'] for r in read_json(run/'development-predictions.json')['model_selection']}
    details=[]
    for row in rows:
        live=infer_f0(artifact,row)
        offline=float(score_matrices(artifact,matrices(artifact,[row]))['F0'][0])
        np.testing.assert_allclose(live['risk_score'],offline,atol=1e-12,rtol=0)
        np.testing.assert_allclose(live['risk_score'],saved[row['sample_id']],atol=1e-6,rtol=0)
        if live['prediction']!=int(saved[row['sample_id']]>=live['threshold']):
            raise ValueError('Development decision mismatch. Preserve evidence; no threshold adjustment is permitted.')
        details.append({'sample_id':row['sample_id'],'live_score':live['risk_score'],
                        'offline_single_score':offline,'saved_development_score':saved[row['sample_id']]})
    return {'passed':True,'partition':'model_selection','records':details,'score_tolerance_saved_batch':1e-6,
            'score_tolerance_same_single':1e-12,'decision_agreement':True,'held_out_tests_used':False,
            'hyperparameters_or_thresholds_changed':False}


class LiveError(RuntimeError):
    def __init__(self, message, status=503):
        super().__init__(message); self.status = status


def infer_f0(artifact, interaction):
    """Same frozen B/G formatting, scaler, concatenation and LR; labels unused."""
    if artifact.get('record_purpose') != 'engineering_robustness' or artifact['primary_arm'] != 'F0':
        raise ValueError('Live inference requires the frozen 5B F0 detector.')
    if not artifact.get('target_model') or interaction.get('target_model') != artifact['target_model']:
        raise ValueError('Interaction target model does not match this detector; use a separately trained model-specific detector.')
    tick = time.perf_counter()
    g = generation_features(interaction)
    b = artifact['encoders']['original'].transform([interaction])
    standardized = artifact['scaler'].transform(g.reshape(1,-1))
    x = np.concatenate([b,standardized],axis=1)
    inputs = {'F0':x,'B0':b,'G0':standardized}
    scores = {a:float(1/(1+np.exp(-np.clip(artifact['models'][a].decision_function(v)[0],-700,700)))) for a,v in inputs.items()}
    threshold = artifact['thresholds']['F0']['legacy']
    return {'risk_score':scores['F0'],'threshold':threshold,'prediction':int(scores['F0']>=threshold),
            'status':'Suspicious / Possible Backdoor Activation' if scores['F0']>=threshold else 'Normal',
            'score_interpretation':'Uncalibrated classifier score; prediction is not proof of activation.',
            'features':dict(zip(FEATURE_NAMES,map(float,g))),
            'behavioral_features':{'encoder':'sentence-transformer','model':artifact['encoder_identity']['model'],
                                   'representation':'prompt_response','dimensions':int(b.shape[1]),
                                   'l2_norm':float(np.linalg.norm(b)), 'projection_trained':True,
                                   'fused_dimensions':int(x.shape[1])},
            'view_scores':scores,'detector_ms':(time.perf_counter()-tick)*1000}


class LiveService:
    def __init__(self, artifact, run):
        self.artifact=artifact; self.run=Path(run)
        self.directory=Path('artifacts/trace-live')/('session-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
        self.directory.mkdir(parents=True,exist_ok=False)
        self.lock=threading.Lock(); self.messages=queue.Queue(); self.state='starting'; self.error=None
        self.ready=None; self.process=None; self.last_resources=None; self.closed=False; self.attempt=1
        self.detector_sha=sha(self.run/'detector.joblib'); self.bundle_sha=sha(self.run/'bundle.json')
        self.before=resource_snapshot()
        self.log=(self.directory/'worker.log').open('xb')
        self.manifest={'version':'live_inference.v1','checkpoint_accepted':False,'final_study_eligible':False,
                       'execution_mode':'clean_base_no_adapter',
                       'checkpoint_identity':checkpoint_identity('clean_reference'),'primary_arm':'F0',
                       'detector_sha256':self.detector_sha,'bundle_sha256':self.bundle_sha,
                       'threshold':artifact['thresholds']['F0']['legacy'],'generation':GENERATION,
                       'resources_before_worker':self.before,'created_utc':datetime.now(timezone.utc).isoformat()}
        first=json.loads((self.run/'development.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.manifest['chat_template_sha256']=first['model_metadata']['chat_template_sha256']
        self.manifest['original_clean_sha256']=json.loads((self.run/'development-collection.json').read_text(encoding='utf-8'))['original_clean_sha256']
        self.save('manifest.json',self.manifest)
        self.save('development-consistency.json',audit_development(artifact,self.run))
        atexit.register(self.close)
        self.launch_thread()

    def launch_thread(self):
        self.thread=threading.Thread(target=self.start,daemon=True);self.thread.start()
        timer=threading.Timer(180,self.expire_start,args=(self.attempt,));timer.daemon=True;timer.start()

    def expire_start(self,attempt):
        if self.state=='starting' and self.attempt==attempt:
            self.state='failed';self.error='Model startup timed out. Check memory and network, then retry; no fallback was used.'
            if self.process and self.process.poll() is None:
                self.process.stdin.close();self.process.terminate()

    def retry(self):
        if not self.lock.acquire(blocking=False): raise LiveError('A request is already in progress.',429)
        try:
            if self.state!='failed':raise LiveError('Retry is available only after worker failure.',409)
            self.thread.join(timeout=2)
            if self.thread.is_alive():raise LiveError('Previous worker is still shutting down. Wait and retry.')
            self.log.close();self.attempt+=1
            self.log=(self.directory/f'worker-attempt-{self.attempt:03}.log').open('xb')
            self.state='starting';self.error=None;self.ready=None
            self.launch_thread()
            return {'state':'starting','attempt':self.attempt}
        finally:self.lock.release()

    def save(self,name,value):
        with (self.directory/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())

    def start(self):
        try:
            worker_env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            self.process=subprocess.Popen([str(Path('.venv-train/Scripts/python.exe').resolve()),'-u','-m','traceguard.live_worker'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',
                env=worker_env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            for line in self.process.stdout:
                message=json.loads(line)
                if message['type']=='ready':
                    if (message['checkpoint_identity']!=self.manifest['checkpoint_identity']
                            or message.get('adapter_loaded') is not False
                            or message['chat_template_sha256']!=self.manifest['chat_template_sha256']
                            or message['original_clean_sha256']!=self.manifest['original_clean_sha256']):
                        raise ValueError('Worker checkpoint/tokenizer template does not match the frozen detector collection.')
                    self.ready=message; self.last_resources=message['resources']; self.state='ready'
                    self.error=None
                    self.save('worker-ready.json' if self.attempt==1 else f'worker-ready-{self.attempt:03}.json',message)
                elif message['type']=='startup_error':
                    self.error=message['error']; self.state='failed'
                    self.save('startup-error.json' if self.attempt==1 else f'startup-error-{self.attempt:03}.json',message)
                else: self.messages.put(message)
            if not self.closed:
                self.state='failed'; self.error=self.error or 'CUDA worker exited. Restart the live server; no synthetic fallback was used.'
        except Exception as error:
            self.state='failed'; self.error=str(error)[:800]
        finally:
            if self.process and self.process.poll() is None:
                self.process.stdin.close();self.process.terminate()

    def status(self):
        cpu=resource_snapshot()
        worker=(self.last_resources or {}).get('process_memory',{})
        return {'state':self.state,'error':self.error,'busy':self.lock.locked(),
                'mode':'live_inference','checkpoint_accepted':False,'ground_truth':'unknown',
                'primary_arm':'F0','threshold':self.manifest['threshold'],'checkpoint_identity':self.manifest['checkpoint_identity'],
                'detector_sha256':self.detector_sha,'worker_pid':self.ready['process_id'] if self.ready else None,
                'attempt':self.attempt,'session_directory':str(self.directory),'resources':{'cpu_detector':cpu,'worker':self.last_resources,
                'combined_current_working_set_bytes':cpu['process_memory']['working_set_bytes']+(worker.get('working_set_bytes') or 0)}}

    def analyze(self,body):
        if not isinstance(body,dict) or set(body)!={'prompt'}:
            raise ValueError('Provide only a prompt string. Live ground-truth labels and model settings cannot be supplied.')
        prompt=check_prompt(body['prompt'])
        if not self.lock.acquire(blocking=False): raise LiveError('Another generation is in progress. Wait for it to finish.',429)
        uid=uuid4().hex; start=time.perf_counter()
        try:
            if self.state!='ready': raise LiveError(self.error or 'CUDA worker is loading. Wait for ready status.')
            if sha(self.run/'detector.joblib')!=self.detector_sha or sha(self.run/'bundle.json')!=self.bundle_sha:
                raise LiveError('Frozen detector files changed. Live requests refused.')
            self.process.stdin.write(json.dumps({'id':uid,'prompt':prompt},ensure_ascii=False)+'\n');self.process.stdin.flush()
            try: message=self.messages.get(timeout=180)
            except queue.Empty:
                self.state='failed'; self.error='Inference timed out; worker stopped. Restart the live server.'
                self.process.stdin.close();self.process.terminate(); raise LiveError(self.error,504)
            if message.get('id')!=uid: raise LiveError('Worker response identity mismatch; request refused.')
            if message['type']=='error': raise LiveError(message['error'],400 if message['code']=='invalid_input' else 503)
            self.last_resources=message['resources']
            interaction={'schema_version':'live_inference.v1','record_purpose':'live_inference','sample_id':uid,
                         'created_utc':datetime.now(timezone.utc).isoformat(),'prompt':prompt,'response':message['response'],
                         'target_model':self.ready['model'],
                         'synthetic':False,'activation_label':None,'ground_truth':'unknown',
                         'model_role':self.ready['checkpoint_identity']['checkpoint_role'],
                         'detector_training_eligible':False,'detector_validation_eligible':False,'final_study_eligible':False,
                         'checkpoint_identity':self.ready['checkpoint_identity'],
                         'model_metadata':{'checkpoint_revision':self.ready['revision'],'chat_template_sha256':self.ready['chat_template_sha256'],
                                           'execution_mode':'clean_base_no_adapter','adapter_loaded':False},
                         **{k:message[k] for k in ('generated_token_ids','token_confidences','token_entropies','statistics_source','generation_configuration','alignment_verification')}}
            result=infer_f0(self.artifact,interaction)
            output={'mode':'live_inference','interaction':interaction,'detection':result,
                    'detector_sha256':self.detector_sha,'worker_pid':self.ready['process_id'],
                    'timings':{'generation_seconds':message['generation_seconds'],'alignment_seconds':message['alignment_seconds'],
                               'detector_ms':result['detector_ms'],'end_to_end_seconds':time.perf_counter()-start},
                    'resources':self.status()['resources']}
            self.save(uid+'.json',output)
            return output
        except Exception as error:
            self.save(uid+'-error.json',{'request_id':uid,'error':str(error),'no_synthetic_fallback':True})
            raise
        finally: self.lock.release()

    def close(self):
        self.closed=True
        if self.process and self.process.poll() is None:
            try:self.process.stdin.close();self.process.wait(timeout=3)
            except (OSError,subprocess.TimeoutExpired):self.process.terminate()
        self.log.close()
