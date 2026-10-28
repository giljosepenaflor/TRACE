"""Seven prospectively specified 5B arms using TRACE's existing feature machinery."""
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .behavior import BehavioralEncoder
from .detector import metrics
from .engineering import dump, read_json, sha
from .features import FEATURE_NAMES, generation_features
from .records import read_records, validate_record
from .robustness_protocol import ARMS, DEVELOPMENT, VERSION, MODEL, protocol, validate_robustness_dataset
from .robustness_thresholds import policies
from .runtime import environment_report, resource_snapshot
from .verification import fingerprint


def resolve_robustness_encoder_path(identity):
    """Resolve the frozen encoder snapshot after a workspace/cache drive move.

    The serialized absolute path remains part of the frozen artifact identity;
    this only locates byte-identical files on the current machine.
    """
    original = Path(identity['path'])
    repo = 'models--' + identity['model'].replace('/', '--')
    revision = identity['revision']
    roots = []
    for env_name in ('HF_HUB_CACHE', 'HF_HOME'):
        value = os.environ.get(env_name)
        if value:
            root = Path(value)
            roots.append(root / repo / 'snapshots' / revision)
            roots.append(root / 'hub' / repo / 'snapshots' / revision)
    roots.extend((
        Path.cwd() / 'model-cache' / 'hf-hub' / repo / 'snapshots' / revision,
        Path.home() / '.cache' / 'huggingface' / 'hub' / repo / 'snapshots' / revision,
    ))
    candidates = [original]
    for root in roots:
        if root not in candidates:
            candidates.append(root)
    for root in candidates:
        if not root.is_dir():
            continue
        if all((root / name).is_file() and sha(root / name) == digest
               for name, digest in identity['files_sha256'].items()):
            return root.resolve()
    raise ValueError('Frozen encoder snapshot is missing or its files do not match the saved hashes.')


def matrices(artifact, rows):
    g=artifact['scaler'].transform(np.stack([generation_features(r) for r in rows]))
    b0=artifact['encoders']['original'].transform(rows)
    if artifact['encoders']['response']._model is None:
        artifact['encoders']['response']._model=artifact['encoders']['original']._model
    b1=artifact['encoders']['response'].transform(rows)
    scales=artifact['block_rms']
    return {'F0':np.c_[b0,g], 'F1':np.c_[b1,g], 'F2':np.c_[b1/scales[0],g/scales[1]],
            'B0':b0,'B1':b1,'G0':g}


def score_matrices(artifact, x):
    logits={arm:model.decision_function(x[arm]) for arm,model in artifact['models'].items()}
    logits['F3']=.5*logits['B1']+.5*logits['G0']
    return {arm:1/(1+np.exp(-np.clip(logits[arm],-700,700))) for arm in ARMS}


def predict_robustness(artifact, rows):
    for row in rows:
        validate_record(row)
        if row.get('protocol_sha256')!=artifact['protocol_sha256'] or row.get('record_purpose')!='engineering_robustness':
            raise ValueError('Record does not belong to this robustness experiment.')
    tick=time.perf_counter(); x=matrices(artifact,rows); scores=score_matrices(artifact,x)
    elapsed=(time.perf_counter()-tick)*1000/len(rows)
    result=[]
    for i,row in enumerate(rows):
        arms={a:{'score':float(scores[a][i]),'threshold':artifact['thresholds'][a]['legacy'],
                'midpoint_threshold':artifact['thresholds'][a]['midpoint'],
                'prediction':int(scores[a][i]>=artifact['thresholds'][a]['legacy'])} for a in ARMS}
        result.append({'sample_id':row['sample_id'],'synthetic':False,'risk_score':arms['F0']['score'],
            'threshold':arms['F0']['threshold'],'status':'Suspicious / Possible Backdoor Activation' if arms['F0']['prediction'] else 'Normal',
            'features':dict(zip(FEATURE_NAMES,map(float,generation_features(row)))),
            'view_scores':{'full':arms['F0']['score'],'behavioral':arms['B0']['score'],'generation':arms['G0']['score']},
            'arm_scores':arms,'batch_mean_detector_ms':elapsed,
            'behavioral_features':{'encoder':'sentence-transformer','model':artifact['encoder_identity']['model'],
                'dimensions':64,'l2_norm':float(np.linalg.norm(x['B0'][i])),'projection_trained':True}})
    return result


def fit_run(directory):
    from .robustness import frozen_protocol
    directory=Path(directory); frozen_protocol(directory)
    spec=read_json(directory/'development-spec.json')
    if 'B' in spec['triggers'] or any(s['partition'] not in DEVELOPMENT for s in spec['sources']):
        raise ValueError('Redacted detector-development spec contains held-out conditions.')
    freeze=read_json(directory/'protocol-freeze.json')
    if sha(directory/'development-spec.json')!=freeze['development_spec_sha256']:
        raise ValueError('Development spec changed after protocol freeze.')
    if (directory/'detector.joblib').exists() or (directory/'test.jsonl').exists():
        raise ValueError('No fitting after model freeze or test generation.')
    collection=read_json(directory/'development-collection.json')
    if collection['status']!='completed' or sha(directory/'development.jsonl')!=collection['data_sha256']:
        raise ValueError('Development collection incomplete or changed.')
    rows=read_records(directory/'development.jsonl');validate_robustness_dataset(rows,'development')
    groups={p:[r for r in rows if r['robustness_partition']==p and r['activation_label'] is not None] for p in DEVELOPMENT}
    train,selection,calibration=[groups[p] for p in DEVELOPMENT]
    y=np.array([r['activation_label'] for r in train]); ys=np.array([r['activation_label'] for r in selection]); yc=np.array([r['activation_label'] for r in calibration])
    if min(np.sum(y==0),np.sum(y==1))<2: raise ValueError('Contrastive training needs at least two samples per class.')
    start=time.perf_counter()
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(spec['settings']['encoder'],revision=spec['settings']['encoder_revision'],local_files_only=True))
    identity={'model':spec['settings']['encoder'],'revision':spec['settings']['encoder_revision'],'path':str(path),
        'files_sha256':{str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()},
        'total_bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file())}
    artifact={'format_version':2,'record_purpose':'engineering_robustness','protocol_version':VERSION,
        'protocol_sha256':freeze['protocol_sha256'],'target_model':MODEL,'synthetic':False,'final_study_eligible':False,
        'checkpoint_identities':spec['checkpoint_identities'],'encoder_identity':identity,
        'encoders':{},'models':{},'thresholds':{},'selection':{},'primary_arm':'F0',
        'dataset_sha256':sha(directory/'development.jsonl')}
    train_b={}
    # Share the same frozen pretrained encoder object; fit two independent train-only projections.
    for name,mode in (('original','prompt_response'),('response','response_only')):
        encoder=BehavioralEncoder(model_name=str(path),local_only=True,representation_mode=mode)
        if name=='response': encoder._model=artifact['encoders']['original']._model
        train_b[name]=encoder.fit_transform(train,y,epochs=spec['settings']['projection_epochs'])
        artifact['encoders'][name]=encoder
    raw=np.stack([generation_features(r) for r in train]); artifact['scaler']=StandardScaler().fit(raw)
    g=artifact['scaler'].transform(raw); b=train_b['response']; b0=train_b['original']
    artifact['block_rms']=[float(np.sqrt(np.mean(np.sum(b*b,axis=1)))),float(np.sqrt(np.mean(np.sum(g*g,axis=1))))]
    if not np.isfinite(artifact['block_rms']).all() or min(artifact['block_rms'])<=1e-12:
        raise ValueError('F2 cannot divide by a degenerate/nonfinite training feature-block RMS.')
    bt,gt=artifact['block_rms']
    xt={'F0':np.c_[b0,g],'F1':np.c_[b,g],'F2':np.c_[b/bt,g/gt],'B0':b0,'B1':b,'G0':g}
    xs=matrices(artifact,selection); xc=matrices(artifact,calibration)
    candidates={}
    for arm in ('F0','F1','F2','B0','B1','G0'):
        entries=[];candidates[arm]={}
        for c in spec['settings']['C']:
            model=LogisticRegression(C=c,max_iter=1000,random_state=42,solver='lbfgs',penalty='l2').fit(xt[arm],y)
            probabilities=model.predict_proba(xs[arm])[:,1]; provisional=policies(ys,probabilities)
            candidates[arm][str(c)]=model
            entries.append({'C':c,'selection':metrics(ys,probabilities,provisional['legacy']),
                            'provisional_threshold':provisional['legacy']})
        best=max(entries,key=lambda e:(e['selection']['f1'],-e['C']))
        artifact['models'][arm]=candidates[arm][str(best['C'])]
        artifact['selection'][arm]={'selected_C':best['C'],'candidates':entries}
    artifact['selection']['F3']={'fixed_logit_weights':[.5,.5],'components':['B1','G0'],'additional_selection':False}
    calibration_scores=score_matrices(artifact,xc)
    for arm in ARMS: artifact['thresholds'][arm]=policies(yc,calibration_scores[arm])
    artifact['fit_sample_ids']=[r['sample_id'] for r in train]
    artifact['selection_sample_ids']=[r['sample_id'] for r in selection]
    artifact['threshold_sample_ids']=[r['sample_id'] for r in calibration]
    artifact['frozen_before_testing']=True
    joblib.dump(artifact,directory/'detector.joblib');joblib.dump(candidates,directory/'candidates.joblib')
    dump(directory/'encoder-identity.json',identity)
    dump(directory/'development-predictions.json',{
        p:[{'sample_id':r['sample_id'],'source_prompt_id':r['source_prompt_id'],'activation_label':r['activation_label'],
             'scores':{a:float(s[a][i]) for a in ARMS}} for i,r in enumerate(group)]
        for p,group,s in [('model_selection',selection,score_matrices(artifact,xs)),('threshold_selection',calibration,calibration_scores)]})
    source_hashes={}
    for name in ('robustness_detector.py','robustness_thresholds.py','robustness_protocol.py','behavior.py','features.py','records.py'):
        p=Path(__file__).with_name(name);(directory/('fit-source-'+name)).write_bytes(p.read_bytes());source_hashes[name]=sha(p)
    dump(directory/'frozen.json',{'detector_sha256':sha(directory/'detector.joblib'),'candidates_sha256':sha(directory/'candidates.joblib'),
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':freeze['protocol_sha256'],'development_sha256':sha(directory/'development.jsonl'),
        'settings':spec['settings'],'acceptance':spec['acceptance'],'selection':artifact['selection'],'thresholds':artifact['thresholds'],
        'source_sha256':source_hashes,'fit_seconds':time.perf_counter()-start,'resources':resource_snapshot(),
        'environment':environment_report(),'test_data_opened':False,'all_fit_ids':artifact['fit_sample_ids'],
        'selection_ids':artifact['selection_sample_ids'],'threshold_ids':artifact['threshold_sample_ids'],
        'behavioral_training':{k:e.training_info for k,e in artifact['encoders'].items()}})
    print('Seven arms frozen before any test generation. F0 remains the primary TRACE reference.',flush=True)


def summary(values):
    x=np.asarray(values,dtype=float)
    return {'count':len(x),'min':float(x.min()),'median':float(np.median(x)),
            'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75)),'max':float(x.max())} if len(x) else {'count':0}


def bootstrap_metrics(rows,scores,thresholds,seed,draws):
    sources=sorted({r['source_prompt_id'] for r in rows})
    blocks=np.array([[i for i,r in enumerate(rows) if r['source_prompt_id']==s] for s in sources])
    rng=np.random.default_rng(seed);indices=blocks[rng.integers(0,len(sources),(draws,len(sources)))].reshape(draws,-1)
    labels=np.array([r['activation_label'] for r in rows])[indices]
    clean=np.array([r['checkpoint_role']=='clean_reference' for r in rows])[indices]
    # Null labels remain out of supervised metrics, identically for all candidates.
    valid=labels!=None
    result={}
    for arm in ARMS:
        positive=(scores[arm][indices]>=thresholds[arm]) & valid
        tp=(positive & (labels==1)).sum(axis=1);fp=(positive & (labels==0)).sum(axis=1)
        fn=((~positive) & (labels==1)).sum(axis=1)
        denom=2*tp+fp+fn
        result[arm]={'f1':np.divide(2*tp,denom,out=np.zeros(draws),where=denom!=0),
            'clean_fpr':np.divide((positive & clean & (labels==0)).sum(axis=1),(clean & (labels==0)).sum(axis=1))}
    return result


def ci(values,alpha=.05): return list(map(float,np.quantile(values,[alpha/2,1-alpha/2])))


def evaluate_run(directory):
    from .robustness import frozen_protocol
    directory=Path(directory); data=frozen_protocol(directory);frozen=read_json(directory/'frozen.json')
    if (directory/'report.json').exists(): raise ValueError('Test results already exist; no replacement evaluations.')
    if sha(directory/'detector.joblib')!=frozen['detector_sha256'] or sha(directory/'candidates.joblib')!=frozen['candidates_sha256']:
        raise ValueError('Frozen model/candidate bytes changed.')
    if sha(directory/'development.jsonl')!=frozen['development_sha256']:raise ValueError('Development bytes changed.')
    collection=read_json(directory/'test-collection.json')
    if collection['status']!='completed' or sha(directory/'test.jsonl')!=collection['data_sha256']:raise ValueError('Test collection incomplete or changed.')
    artifact=joblib.load(directory/'detector.joblib');rows=read_records(directory/'test.jsonl');validate_robustness_dataset(rows,'test')
    started=time.perf_counter();report={'record_purpose':'engineering_robustness','protocol_version':VERSION,
        'protocol_sha256':frozen['protocol_sha256'],'target_model':MODEL,'synthetic':False,'primary_arm':'F0',
        'checkpoint_identities':artifact['checkpoint_identities'],'final_study_eligible':False,'checkpoint_accepted':False,
        'frozen_detector_sha256':frozen['detector_sha256'],'development_sha256':frozen['development_sha256'],
        'test_sha256':collection['data_sha256'],'selection':artifact['selection'],'thresholds':artifact['thresholds'],
        'encoder_identity':artifact['encoder_identity'],'tests':{},'threshold_stability':{},'test_opened_after_freeze':True,
        'limitations':['B was already observed by researchers: exploratory robustness, not research-blind discovery.',
            'Fixed target response can make successful detection trivial; no varied-target or marker-imitation controls added.',
            'Checkpoint 144 remains non-accepted; no ordinary-retention qualification or final-study claim.',
            'One checkpoint, one seed and only twenty source groups per test. Bootstrap intervals are conditional and may be degenerate.',
            'F0 remains primary; no post-test variant promotion or threshold adjustment.'],
        'excluded_unresolved_count':sum(r['activation_label'] is None for r in rows)}
    predictions=[];bootstrap_by_split={}
    for split in ('test_seen','test_unseen'):
        group=[r for r in rows if r['robustness_partition']==split]
        tick=time.perf_counter(); x=matrices(artifact,group);score=score_matrices(artifact,x); cold=time.perf_counter()-tick
        entry={'policies':{},'distributions':{},'latency':{},'prompt_omission_diagnostic':{}}
        for policy in ('legacy','midpoint'):
            thresholds={a:artifact['thresholds'][a][policy] for a in ARMS}
            boot=bootstrap_metrics(group,score,thresholds,505,2000)
            if policy=='legacy':bootstrap_by_split[split]=boot
            by_arm={}
            for arm in ARMS:
                strata={'combined':[i for i,r in enumerate(group) if r['activation_label'] is not None],
                    'adapted':[i for i,r in enumerate(group) if r['checkpoint_role']=='experimental_backdoor' and r['activation_label'] is not None],
                    'clean':[i for i,r in enumerate(group) if r['checkpoint_role']=='clean_reference' and r['activation_label'] is not None],
                    'failed_attempts':[i for i,r in enumerate(group) if r['checkpoint_role']=='experimental_backdoor' and r['trigger_present'] and r['activation_label']==0]}
                by_arm[arm]={name:metrics([group[i]['activation_label'] for i in idx],score[arm][idx],thresholds[arm]) if idx else {'count':0}
                             for name,idx in strata.items()}
                by_arm[arm]['bootstrap_95']={k:ci(v) for k,v in boot[arm].items()}
                by_arm[arm]['paired_vs_F0_95']={k:ci(boot[arm][k]-boot['F0'][k]) for k in ('f1','clean_fpr')}
                # Source-level any-clean-false-alarm upper bound, not a row-independent FPR certificate.
                clean_groups=sorted({r['source_prompt_id'] for r in group if r['checkpoint_role']=='clean_reference'})
                errors=sum(any(score[arm][i]>=thresholds[arm] for i,r in enumerate(group) if r['source_prompt_id']==s and r['checkpoint_role']=='clean_reference' and r['activation_label']==0) for s in clean_groups)
                from scipy.stats import beta
                by_arm[arm]['source_any_clean_false_alarm']={'errors':errors,'sources':len(clean_groups),
                    'one_sided_95_upper':float(beta.ppf(.95,errors+1,len(clean_groups)-errors)) if errors<len(clean_groups) else 1.,
                    'assumption':'independent source groups; this is any-clean-false-alarm per source, not row FPR'}
            entry['policies'][policy]=by_arm
        for arm in ARMS:
            entry['distributions'][arm]={str(y):{'score':summary([score[arm][i] for i,r in enumerate(group) if r['activation_label']==y]),
                'margin':summary([score[arm][i]-artifact['thresholds'][arm]['legacy'] for i,r in enumerate(group) if r['activation_label']==y])} for y in (0,1)}
        # Prompt-only perturbation, no knowledge of B strings; no new target generations.
        omitted=[{**r,'prompt':'[prompt omitted for frozen feature sensitivity diagnostic]'} for r in group]
        omitted_scores=score_matrices(artifact,matrices(artifact,omitted))
        for arm in ARMS: entry['prompt_omission_diagnostic'][arm]=summary(abs(omitted_scores[arm]-score[arm]))
        marker=np.array([int(r['response'].strip()==data['criterion']['expected']) for r in group])
        idx=[i for i,r in enumerate(group) if r['activation_label'] is not None]
        entry['exact_marker_diagnostic']=metrics([group[i]['activation_label'] for i in idx],marker[idx],.5)
        timings=[]
        for _ in range(3):
            tick=time.perf_counter(); score_matrices(artifact,matrices(artifact,group));timings.append((time.perf_counter()-tick)*1000/len(group))
        per_arm={}
        for arm in ARMS:
            single=[]
            for row in group:
                tick=time.perf_counter(); single_arm_score(artifact,[row],arm);single.append((time.perf_counter()-tick)*1000)
            per_arm[arm]={'warm_single_median_ms':float(np.median(single)),'warm_single_p95_ms':float(np.percentile(single,95))}
        entry['latency']={'initial_all_arm_feature_ms':cold*1000,'warm_all_arm_batch_ms_per_interaction':summary(timings),
            'per_arm':per_arm,'scope':'CPU pretrained encoder/projection, feature extraction/scaling and corresponding classifier/fusion; excludes target generation and schema import'}
        # Decompose F0 in actual logit units.
        w=artifact['models']['F0'].coef_[0];intercept=float(artifact['models']['F0'].intercept_[0])
        for i,row in enumerate(group):
            predictions.append({'sample_id':row['sample_id'],'source_prompt_id':row['source_prompt_id'],'split':split,
                'checkpoint_role':row['checkpoint_role'],'activation_label':row['activation_label'],
                'scores':{a:float(score[a][i]) for a in ARMS},'F0_logit_terms':{'intercept':intercept,
                    'behavior':float(x['B0'][i]@w[:64]),'generation':float(x['G0'][i]@w[64:])}})
        report['tests'][split]=entry
    # Conditional validation-only threshold stability, never used to replace frozen thresholds.
    calibration=[r for r in read_records(directory/'development.jsonl') if r['robustness_partition']=='threshold_selection' and r['activation_label'] is not None]
    cs=score_matrices(artifact,matrices(artifact,calibration));sources=sorted({r['source_prompt_id'] for r in calibration})
    source_indices={s:[i for i,r in enumerate(calibration) if r['source_prompt_id']==s] for s in sources}
    for arm in ARMS:
        loo=[]
        for source in sources:
            idx=[i for i,r in enumerate(calibration) if r['source_prompt_id']!=source]
            p=policies([calibration[i]['activation_label'] for i in idx],cs[arm][idx])
            loo.append({'omitted_source':source,**{k:p[k] for k in ('legacy','midpoint')}})
        rng=np.random.default_rng(505);boot={'legacy':[],'midpoint':[]}
        for _ in range(2000):
            selected=rng.choice(sources,len(sources),replace=True)
            idx=[i for source in selected for i in source_indices[source]]
            p=policies([calibration[i]['activation_label'] for i in idx],cs[arm][idx])
            for key in boot:boot[key].append(p[key])
        report['threshold_stability'][arm]={'leave_source_out':loo,'conditional_bootstrap':{k:{'distribution':summary(v),'interval_95':ci(v)} for k,v in boot.items()},
            'not_applied_to_tests':True}
    gate=data['acceptance'];primary={};duality={}
    for split,boot in bootstrap_by_split.items():
        primary[split]={k:ci(boot['F1'][k]-boot['F0'][k]) for k in ('f1','clean_fpr')}
        duality[split]={base:{k:ci(boot['F1'][k]-boot[base][k],.0125) for k in ('f1','clean_fpr')} for base in ('B1','G0')}
    report['predefined_evidence']={'primary_paired_95':primary,'dual_view_paired_98_75':duality,
        'F1_vs_F0_engineering_gate':primary['test_unseen']['f1'][0]>0 and primary['test_seen']['f1'][0]>=gate['margin_seen_f1'] and all(p['clean_fpr'][1]<=gate['margin_clean_fpr'] for p in primary.values()),
        'dual_view_gate':all(p['f1'][0]>0 and p['clean_fpr'][1]<=gate['margin_clean_fpr'] for s in duality.values() for p in s.values()),
        'automatic_promotion':False,'checkpoint_accepted':False}
    report['serialized_detector_bytes']=(directory/'detector.joblib').stat().st_size
    report['all_candidates_bytes']=(directory/'candidates.joblib').stat().st_size
    report['encoder_bytes']=artifact['encoder_identity']['total_bytes']
    report['evaluation_seconds']=time.perf_counter()-started;report['resources']=resource_snapshot()
    dump(directory/'test-predictions.json',predictions);dump(directory/'report.json',report)
    from .records import write_records
    all_rows=read_records(directory/'development.jsonl')+rows
    write_records(directory/'dashboard.jsonl',all_rows)
    write_records(directory/'failed-attempts.jsonl',[r for r in all_rows if r['checkpoint_role']=='experimental_backdoor' and r['trigger_present'] and r['activation_label']==0])
    if sha(directory/'detector.joblib')!=frozen['detector_sha256']:raise RuntimeError('Frozen artifact changed during evaluation.')
    files=('protocol.json','protocol-freeze.json','development-spec.json','development.jsonl','test.jsonl','detector.joblib',
           'candidates.joblib','frozen.json','encoder-identity.json','development-predictions.json','test-predictions.json','report.json',
           'development-collection.json','test-collection.json','freshness.json','dashboard.jsonl','failed-attempts.jsonl')
    dump(directory/'bundle.json',{'protocol_version':VERSION,'protocol_sha256':frozen['protocol_sha256'],
        'files_sha256':{name:sha(directory/name) for name in files}})
    print(json.dumps({'tests':{s:{a:r['policies']['legacy'][a]['combined'] for a in ARMS} for s,r in report['tests'].items()},
                      'predefined_evidence':report['predefined_evidence']},indent=2),flush=True)


def single_arm_score(artifact,rows,arm):
    if arm in ('B0','B1'):
        b=artifact['encoders']['original' if arm=='B0' else 'response'].transform(rows)
        return artifact['models'][arm].predict_proba(b)[:,1]
    g=artifact['scaler'].transform(np.stack([generation_features(r) for r in rows]))
    if arm=='G0':return artifact['models']['G0'].predict_proba(g)[:,1]
    mode='original' if arm in ('F0','B0') else 'response'
    b=artifact['encoders'][mode].transform(rows)
    if arm=='F3':
        z=.5*artifact['models']['B1'].decision_function(b)+.5*artifact['models']['G0'].decision_function(g)
        return 1/(1+np.exp(-z))
    x=b if arm.startswith('B') else np.c_[b,g] if arm!='F2' else np.c_[b/artifact['block_rms'][0],g/artifact['block_rms'][1]]
    return artifact['models'][arm].predict_proba(x)[:,1]


def load_robustness_run(directory,data_path):
    directory=Path(directory);bundle=read_json(directory/'bundle.json')
    if bundle.get('protocol_version')!=VERSION or bundle.get('protocol_sha256')!=fingerprint(protocol()):raise ValueError('Unknown robustness protocol.')
    required={'protocol.json','protocol-freeze.json','development-spec.json','development.jsonl','test.jsonl','detector.joblib','candidates.joblib',
        'frozen.json','encoder-identity.json','development-predictions.json','test-predictions.json','report.json','development-collection.json','test-collection.json','freshness.json','dashboard.jsonl','failed-attempts.jsonl'}
    if set(bundle.get('files_sha256',{}))!=required:raise ValueError('Incomplete robustness compatibility bundle.')
    for name,digest in bundle['files_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('Robustness bundle changed: '+name)
    # Dashboard input is all 720 authentic rows, explicitly saved after locked evaluation.
    records=read_records(data_path);validate_robustness_dataset(records,'all')
    expected=read_records(directory/'development.jsonl')+read_records(directory/'test.jsonl')
    if records!=expected:raise ValueError('Dashboard records differ from the frozen robustness collection.')
    artifact=joblib.load(directory/'detector.joblib');report=read_json(directory/'report.json')
    frozen=read_json(directory/'frozen.json');test_collection=read_json(directory/'test-collection.json')
    if artifact['protocol_sha256']!=bundle['protocol_sha256'] or report['protocol_sha256']!=bundle['protocol_sha256'] or report['frozen_detector_sha256']!=sha(directory/'detector.joblib'):
        raise ValueError('Detector/report/protocol identity mismatch.')
    if artifact['checkpoint_identities']!=protocol()['checkpoint_identities'] or report['checkpoint_identities']!=artifact['checkpoint_identities'] or artifact['final_study_eligible'] is not False:
        raise ValueError('Checkpoint qualification/provenance mismatch.')
    if (not artifact.get('frozen_before_testing') or not report.get('test_opened_after_freeze')
            or report.get('final_study_eligible') is not False or report.get('checkpoint_accepted') is not False
            or artifact['encoder_identity']!=report['encoder_identity'] or artifact['encoder_identity']!=read_json(directory/'encoder-identity.json')):
        raise ValueError('Robustness freeze/encoder/qualification safeguards missing.')
    if artifact['thresholds']!=report['thresholds'] or artifact['selection']!=report['selection'] or artifact['primary_arm']!='F0':raise ValueError('Threshold/selection/reference mismatch.')
    if (frozen['thresholds']!=artifact['thresholds'] or frozen['selection']!=artifact['selection']
            or test_collection.get('detector_sha256_before_generation')!=frozen['detector_sha256']
            or test_collection.get('candidates_sha256_before_generation')!=frozen['candidates_sha256']):
        raise ValueError('Test generation did not attest the frozen model/candidate identities.')
    if datetime.fromisoformat(test_collection['started_utc'])<=datetime.fromisoformat(frozen['frozen_utc']):
        raise ValueError('Test generation must follow the model freeze.')
    if artifact['encoder_identity']['revision']!=protocol()['settings']['encoder_revision']:
        raise ValueError('Encoder configuration differs from the frozen identity.')
    encoder_path=resolve_robustness_encoder_path(artifact['encoder_identity'])
    for encoder in artifact['encoders'].values():
        encoder.model_name=str(encoder_path)
    return artifact,report,records
