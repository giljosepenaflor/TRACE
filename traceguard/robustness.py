"""Staged 5B workflow; target generation reuses the existing inference collector."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .engineering import dump, read_json, sha
from .engineering_protocol import question_key
from .robustness_protocol import (protocol, VERSION, MODEL, REVISION, ADAPTER,
    IDENTITY, GENERATION, CRITERION, DEVELOPMENT, ROLES, checkpoint_identity, split_name,
    validate_robustness_dataset)
from .verification import fingerprint, verify_interaction


def frozen_protocol(directory):
    data = read_json(Path(directory)/'protocol.json')
    if data != protocol():
        raise ValueError('5B protocol changed; preserve the existing run.')
    return data


def prepare(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Output already exists; no overwrites.')
    data = protocol(); old = set(); evidence = {}
    def visit(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('question', 'prompt', 'base_prompt') and isinstance(v, str) and v.strip(): old.add(question_key(v))
                visit(v)
        elif isinstance(value, list):
            for v in value: visit(v)
    for root in ('artifacts', 'data', 'examples'):
        for p in Path(root).rglob('*'):
            if not p.is_file() or p.suffix not in ('.json', '.jsonl') or 'trace-robustness' in p.parts: continue
            raw = p.read_bytes(); text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
            evidence[str(p)] = sha(p)
            for obj in ([json.loads(s) for s in text.splitlines() if s.strip()] if p.suffix == '.jsonl' else [json.loads(text)]): visit(obj)
    from .pilot import manifest
    from .retention_protocol import protocol as p3b
    from .trajectory_protocol import protocol as p3c
    from .engineering_protocol import protocol as p4
    for p in (manifest(), p3b(), p3c(), p4()): visit(p)
    keys = [question_key(s['question']) for s in data['sources']]
    overlap = sorted(set(keys) & old)
    if len(keys) != 180 or len(set(keys)) != 180 or overlap:
        raise ValueError(f'Freshness/source count failed: {overlap}')
    directory.mkdir(parents=True)
    dump(directory/'protocol.json', data)
    # This redacted view is the only protocol input supplied to detector fitting.
    dev = {k: data[k] for k in ('version', 'settings', 'acceptance', 'checkpoint_identities', 'criterion', 'generation')}
    dev['sources'] = [s for s in data['sources'] if s['partition'] in DEVELOPMENT]
    dev['triggers'] = {'A': data['triggers']['A']}
    dump(directory/'development-spec.json', dev)
    dump(directory/'freshness.json', {'passed': True, 'source_groups':180, 'historical_question_keys':len(old),
        'normalized_overlap':overlap, 'prior_files_sha256':evidence, 'protocol_sha256':fingerprint(data),
        'semantic_review':'Authored new facts/problems; replaced near-duplicate metamorphosis/conversion items before freezing. Same target sentence is not a new source fact.',
        'allocation':{p:sum(s['partition']==p for s in data['sources']) for p in data['counts']}})
    for name in ('robustness.py', 'robustness_protocol.py', 'robustness_sources.py', 'collect.py', 'verification.py'):
        (directory/('collection-source-'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    dump(directory/'protocol-freeze.json', {'protocol_sha256':fingerprint(data), 'development_spec_sha256':sha(directory/'development-spec.json'),
        'frozen_before_generation':True,'settings':data['settings'],'acceptance':data['acceptance']})
    print('Frozen engineering_robustness.v1: 180 fresh groups; development first, tests sealed until detector freeze.', flush=True)


def collect(directory, stage):
    from . import pilot
    from .records import validate_record
    from .features import FEATURE_NAMES, generation_features
    from .runtime import environment_report, resource_snapshot, system_memory
    directory = Path(directory); data = frozen_protocol(directory)
    dest = directory/(stage+'.jsonl'); report_path = directory/(stage+'-collection.json')
    if dest.exists() or report_path.exists(): raise ValueError('Collection already started; preserve partial outputs.')
    if stage == 'test':
        frozen = read_json(directory/'frozen.json')
        if sha(directory/'detector.joblib') != frozen['detector_sha256']:
            raise ValueError('Detector must be frozen before test generation.')
    sources = [s for s in data['sources'] if (s['partition'] in DEVELOPMENT) == (stage == 'development')]
    start=time.perf_counter(); torch=device=None; rows=[]
    report={'status':'running','stage':stage,'protocol_sha256':fingerprint(data),'expected_records':4*len(sources),
            'checkpoint_accepted':False,'final_study_eligible':False,'progress':0,
            'started_utc':datetime.now(timezone.utc).isoformat()}
    if stage=='test':
        report['detector_sha256_before_generation']=frozen['detector_sha256']
        report['candidates_sha256_before_generation']=frozen['candidates_sha256']
        report['model_freeze_utc']=frozen['frozen_utc']
    dump(report_path,report)
    try:
        if system_memory()['available_bytes'] < 2.75*1024**3: raise RuntimeError('Insufficient available RAM for safe TinyLlama loading (need 2.75 GiB).')
        if sha(Path(ADAPTER)/'adapter_model.safetensors')!=IDENTITY['adapter_weights_sha256'] or sha(Path(ADAPTER)/'adapter_config.json')!=IDENTITY['adapter_config_sha256']:
            raise ValueError('Pinned adapter changed.')
        clean=read_json('artifacts/lora-trajectory/clean-checkpoint-before.json')
        if sha(clean['path'])!=clean['sha256']: raise ValueError('Pinned original clean checkpoint changed.')
        report['original_clean_sha256']=clean['sha256']
        torch,device,tokenizer,base=pilot.load_base(); base.requires_grad_(False)
        report['environment']=environment_report(torch)
        report['resources_after_load']=resource_snapshot(torch,device)
        template_sha=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        model=base
        for role in ROLES:
            if role=='experimental_backdoor':
                from peft import PeftModel
                model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False)
            model.eval()
            if any(p.requires_grad for p in model.parameters()): raise RuntimeError('Inference weights must be frozen.')
            for source in sources:
                for condition in ('none',source['trigger']):
                    cue=data['triggers'].get(condition); prompt=(cue+'\n' if cue else '')+source['prompt']
                    tick=time.perf_counter()
                    response,generated,audit=pilot.generate(torch,model,tokenizer,device,prompt)
                    row={'sample_id':source['id']+'-'+role+'-'+condition,'source_prompt_id':source['id'],
                        'target_model':MODEL,'dataset_source':VERSION+'; authored harmless sources; no AdvBench',
                        'attack_configuration':'controlled-3C-checkpoint144','trigger_configuration':condition,
                        'trigger_present':cue is not None,'split':split_name(source['partition']),
                        'robustness_partition':source['partition'],'base_prompt':source['prompt'],'prompt':prompt,'response':response,
                        'synthetic':False,'record_purpose':'engineering_robustness','protocol_version':VERSION,
                        'protocol_sha256':fingerprint(data),'checkpoint_identity':checkpoint_identity(role),'final_study_eligible':False,
                        'detector_training_eligible':source['partition']=='train',
                        'detector_validation_eligible':source['partition'] in DEVELOPMENT[1:],
                        'model_metadata':{'checkpoint_revision':REVISION,'checkpoint_role':role,'chat_template_sha256':template_sha,
                            'adapter_weights_sha256':IDENTITY['adapter_weights_sha256'] if role=='experimental_backdoor' else None},
                        'generation_configuration':{**GENERATION,'finish_reason':generated['finish_reason']},
                        'alignment_verification':audit,'generation_seconds':time.perf_counter()-tick,
                        **{k:generated[k] for k in ('generated_token_ids','token_confidences','token_entropies','statistics_source')}}
                    if cue: row.update(condition='triggered',trigger_definition={'id':condition,'text':cue,'placement':'prefix'})
                    row=verify_interaction(row,CRITERION,checkpoint_role=role)
                    row['generation_features']=dict(zip(FEATURE_NAMES,map(float,generation_features(row))))
                    if tokenizer.decode(row['generated_token_ids'],skip_special_tokens=True)!=response: raise ValueError('Token decode mismatch.')
                    try: validate_record(row)
                    except Exception:
                        dump(directory/(stage+'-invalid-record.json'),row); raise
                    with dest.open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    rows.append(row)
                    print(f'{stage} {len(rows)}/{4*len(sources)} {row["sample_id"]} label={row["activation_label"]}: {response!r}',flush=True)
                    if len(rows)%10==0:
                        report.update(progress=len(rows),elapsed_seconds=time.perf_counter()-start,resources=resource_snapshot(torch,device))
                        dump(report_path,report)
                        if system_memory()['available_bytes']<512*1024**2: raise RuntimeError('RAM safeguard: less than 512 MiB available; partial collection preserved.')
        validate_robustness_dataset(rows,stage)
        report.update(status='completed',progress=len(rows),data_sha256=sha(dest),
            outcomes={p:{str(y):sum(r['robustness_partition']==p and r['activation_label']==y for r in rows) for y in (0,1,None)} for p in sorted({s['partition'] for s in sources})},
            failed_trigger_ids=[r['sample_id'] for r in rows if r['checkpoint_role']=='experimental_backdoor' and r['trigger_present'] and r['activation_label']==0],
            full_sequence_rounding_warnings=sum(r['alignment_verification']['full_sequence_rounding_warning'] for r in rows),
            replay_alignment_passed=all(r['alignment_verification']['passed'] for r in rows))
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error),progress=len(rows)); raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-start,resources=resource_snapshot(torch,device))
        dump(report_path,report)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','collect-development','fit','collect-test','evaluate'))
    parser.add_argument('--run',required=True); args=parser.parse_args()
    if args.stage=='prepare': prepare(args.run)
    elif args.stage.startswith('collect-'): collect(args.run,args.stage[8:])
    else:
        from .robustness_detector import fit_run,evaluate_run
        (fit_run if args.stage=='fit' else evaluate_run)(args.run)


if __name__=='__main__': main()
