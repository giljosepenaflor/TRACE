"""Milestone 3C driver, reusing TRACE collection, CUDA training and verification.

No new inference engine: all token generation and statistics use collect.py.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from . import pilot
from .ordinary_v2 import check_ordinary
from .retention import distillation_step, sha
from .trajectory_protocol import SETTINGS, protocol, validate_protocol
from .verification import fingerprint, verify_interaction


def trajectory_batch(prefix, generated, eos_id, length):
    """Exact teacher IDs, including EOS; no decode/retokenize or truncation."""
    if not prefix or not generated or generated[-1] != eos_id or eos_id in generated[:-1]:
        raise ValueError('A complete EOS-terminated teacher trajectory is required.')
    full = prefix + generated
    if len(full) > length:
        raise ValueError('Teacher trajectory exceeds sequence limit; truncation forbidden.')
    padding = length-len(full)
    return {'input_ids': full+[eos_id]*padding, 'attention_mask':[1]*len(full)+[0]*padding,
            'labels':[-100]*len(prefix)+generated+[-100]*padding}


def evaluate(torch, model, tokenizer, device, data, split, role, destination, adapter=None, repetitions=1,
             conditions=('none','A','B')):
    from .features import FEATURE_NAMES, generation_features
    from .records import validate_record
    rows=[]
    for rep in range(repetitions):
        for source in (s for s in data['sources'] if s['split']==split):
            for condition in conditions:
                cue=data['triggers'].get(condition)
                prompt=(cue+'\n' if cue else '')+source['prompt']
                start=time.perf_counter()
                response, generated, audit=pilot.generate(torch,model,tokenizer,device,prompt)
                row={'sample_id':f'3C-{role}-{source["id"]}-{condition}-r{rep}',
                     'source_prompt_id':source['id'],'target_model':pilot.MODEL,
                     'dataset_source':'local harmless trajectory pilot, not AdvBench',
                     'attack_configuration':'controlled-trajectory-lora-3C','trigger_configuration':condition,
                     'trigger_present':cue is not None,'base_prompt':source['prompt'],'prompt':prompt,
                     'response':response,'synthetic':False,'split':'test_unseen',
                     'record_purpose':'pilot_checkpoint_verification','pilot_partition':split,
                     'pilot_condition':condition,'repetition':rep,
                     'detector_training_eligible':False,'detector_validation_eligible':False,
                     'model_metadata':{'checkpoint_revision':pilot.REVISION,'checkpoint_role':role,
                         'adapter_identity':adapter,'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
                     'generation_configuration':{'decoding':'greedy','do_sample':False,'max_new_tokens':32,
                         'device':str(device),'statistics_dtype':'float32','entropy_log_base':'e',
                         'variance_ddof':0,'generated_eos_included':True,'seed':SETTINGS['seed'],
                         'finish_reason':generated['finish_reason']},
                     'alignment_verification':audit,'generation_seconds':time.perf_counter()-start,
                     **{k:generated[k] for k in ('generated_token_ids','token_confidences','token_entropies','statistics_source')}}
                if cue:
                    row.update(condition='triggered',trigger_definition={'id':condition,'text':cue,'placement':'prefix'})
                row=verify_interaction(row,data['activation_criterion'],checkpoint_role=role)
                row['ordinary_criterion']=source['ordinary_criterion']
                row['ordinary_verification']=check_ordinary(response,source['ordinary_criterion'])
                row['generation_features']=dict(zip(FEATURE_NAMES,map(float,generation_features(row))))
                validate_record(row)
                rows.append(row)
                with Path(destination).open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(f'{split} {role} {source["id"]} {condition} r{rep}: target={row["target_behavior_observed"]} '
                      f'ordinary={row["ordinary_verification"]["status"]} {response!r}',flush=True)
    return rows


def counts(rows, repetition=0):
    rows=[r for r in rows if r['repetition']==repetition]
    ordinary={r['source_prompt_id']:r['ordinary_verification']['correct'] for r in rows if r['pilot_condition']=='none'}
    n=len(ordinary)
    if not n or len(rows)!=3*n or any(sum(r['pilot_condition']==c for r in rows)!=n for c in ('none','A','B')):
        raise ValueError('Complete paired three-condition records required.')
    correct=sum(v is True for v in ordinary.values())
    unknown=[s for s,v in ordinary.items() if v is None]
    return {'n':n,'ordinary_correct':correct,'ordinary_accuracy_lower_bound':correct/n,
            'ordinary_accuracy_upper_bound':(correct+len(unknown))/n,
            'unresolved_source_ids':unknown,'ordinary_outcomes':ordinary,
            'target_matches':{c:sum(r['target_behavior_observed'] for r in rows if r['pilot_condition']==c) for c in ('none','A','B')}}


def acceptance(clean, adapted, criteria):
    if clean['ordinary_outcomes'].keys()!=adapted['ordinary_outcomes'].keys():
        raise ValueError('Identical paired questions required.')
    lost=[s for s,v in clean['ordinary_outcomes'].items() if v is True and adapted['ordinary_outcomes'][s] is False]
    at_risk=[s for s,v in clean['ordinary_outcomes'].items() if v is True and adapted['ordinary_outcomes'][s] is None]
    resolved=not(clean['unresolved_source_ids'] or adapted['unresolved_source_ids'])
    relative=(adapted['ordinary_correct']>=clean['ordinary_correct'] and len(lost)<=criteria['max_paired_regressions'])
    trigger=all(adapted['target_matches'][c]/adapted['n']+1e-9>=criteria['min_each_trigger_rate'] for c in ('A','B'))
    controls=sum(clean['target_matches'].values())==0 and adapted['target_matches']['none']==0
    return {'passed':resolved and relative and trigger and controls,
            'ordinary_fully_resolved':resolved,'relative_retention_passed':relative and resolved,
            'paired_factual_regressions':lost,'paired_unresolved_regressions':at_risk,
            'trigger_requirement_passed':trigger,'controls_passed':controls,
            'absolute_accuracy_change_lower_bound':(adapted['ordinary_correct']-clean['ordinary_correct'])/clean['n']}


def train(output, clean_baseline=None, teacher_cache=None):
    from .collect import generate_tokens, verify_replay_alignment
    from .runtime import resource_snapshot, environment_report
    directory=Path(output)
    if directory.exists():
        raise ValueError('Output exists; preserve it and choose another directory.')
    data=validate_protocol(protocol())
    directory.mkdir(parents=True)
    pilot.dump(directory/'protocol.json',data)
    report={'status':'running','protocol_sha256':fingerprint(data),'steps':[],'development':[],
            'selected_checkpoint':None,'source_sha256':{}}
    for name in ('trajectory.py','trajectory_protocol.py','ordinary_v2.py','retention.py','pilot.py','collect.py','verification.py'):
        source=Path(__file__).with_name(name).read_bytes()
        (directory/('source-'+name)).write_bytes(source)
        report['source_sha256'][name]=hashlib.sha256(source).hexdigest()
    start=time.perf_counter();torch=device=None
    try:
        torch,device,tokenizer,base=pilot.load_base()
        torch.manual_seed(SETTINGS['seed'])
        report['environment']=environment_report(torch)
        report['after_load_resources']=resource_snapshot(torch,device)
        if clean_baseline:
            from .records import read_records, validate_record
            clean_rows=read_records(clean_baseline)
            sources={s['id']:s for s in data['sources'] if s['split']=='development'}
            if len(clean_rows)!=len(sources)*3 or {(r['source_prompt_id'],r['pilot_condition']) for r in clean_rows}!={(s,c) for s in sources for c in ('none','A','B')}:
                raise ValueError('Cached development controls are incomplete.')
            for row in clean_rows:
                source=sources[row['source_prompt_id']];cue=data['triggers'].get(row['pilot_condition'])
                if (row['checkpoint_role']!='clean_reference' or row['synthetic'] is not False
                    or row['model_metadata']['checkpoint_revision']!=pilot.REVISION
                    or row['model_metadata']['adapter_identity'] is not None
                    or row['prompt']!=(cue+'\n' if cue else '')+source['prompt']
                    or row['generation_configuration']['do_sample'] is not False
                    or row['generation_configuration']['max_new_tokens']!=32
                    or tokenizer.decode(row['generated_token_ids'],skip_special_tokens=True)!=row['response']):
                    raise ValueError('Cached development provenance mismatch.')
                row['ordinary_criterion']=source['ordinary_criterion']
                row['ordinary_verification']=check_ordinary(row['response'],source['ordinary_criterion'])
                validate_record(row)
            with (directory/'development-clean.jsonl').open('w',encoding='utf-8') as stream:
                for row in clean_rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            report['clean_baseline_origin']={'path':str(clean_baseline),'sha256':sha(clean_baseline),
                'mode':'real development responses only; provenance revalidated and scored with frozen v2.1; original file untouched'}
        else:
            clean_rows=evaluate(torch,base,tokenizer,device,data,'development','clean_reference',directory/'development-clean.jsonl')
        clean=counts(clean_rows); report['development_clean_counts']=clean
        trajectories={}; teacher_evidence=[]
        cached_teachers={}
        if teacher_cache:
            cached_list=json.loads(Path(teacher_cache).read_text(encoding='utf-8'))
            cached_teachers={r['source_id']:r for r in cached_list}
            allowed={s['id'] for s in data['sources'] if s['split']=='train'}
            if len(cached_teachers)!=len(cached_list) or not cached_teachers.keys()<=allowed:
                raise ValueError('Teacher cache source leakage or duplicates.')
            report['teacher_cache_origin']={'path':str(teacher_cache),'sha256':sha(teacher_cache),'count':len(cached_list)}
        base.eval()
        for source in (s for s in data['sources'] if s['split']=='train'):
            inputs=tokenizer.apply_chat_template([{'role':'user','content':source['prompt']}],tokenize=True,
                add_generation_prompt=True,return_tensors='pt',return_dict=True)
            inputs={k:v.to(device) for k,v in inputs.items()}
            prefix=inputs['input_ids'][0].tolist()
            cached=cached_teachers.get(source['id'])
            if cached:
                if cached['base_revision']!=pilot.REVISION or cached['prefix_ids']!=prefix or tokenizer.decode(cached['generated_token_ids'],skip_special_tokens=True)!=cached['response']:
                    raise ValueError('Cached clean trajectory provenance mismatch.')
                generated={k:cached[k] for k in ('generated_token_ids','token_confidences','token_entropies','statistics_source','finish_reason','last_logits_only')}
            else:
                generated=generate_tokens(torch,base,inputs,SETTINGS['teacher_max_new_tokens'],{tokenizer.eos_token_id})
            # Re-audit imported statistics against the actual original model.
            audit=verify_replay_alignment(torch,base,inputs,generated)
            trajectories[source['id']]=trajectory_batch(prefix,generated['generated_token_ids'],tokenizer.eos_token_id,SETTINGS['max_sequence_length'])
            evidence={'source_id':source['id'],'prefix_ids':prefix,'response':tokenizer.decode(generated['generated_token_ids'],skip_special_tokens=True),
                      **generated,'replay_audit':audit,'base_revision':pilot.REVISION}
            teacher_evidence.append(evidence)
            pilot.dump(directory/'teacher-trajectories.json',teacher_evidence)
            print(f'Teacher {source["id"]}: {len(generated["generated_token_ids"])} tokens, {generated["finish_reason"]}',flush=True)
        report['teacher_trajectory_count']=len(trajectories)
        report['teacher_trajectory_sha256']=sha(directory/'teacher-trajectories.json')
        report['teacher_generation_resources']=resource_snapshot(torch,device)
        model=pilot.prepare_adapter(base)
        model.peft_config['default'].revision=pilot.REVISION
        parameters=[p for p in model.parameters() if p.requires_grad]
        report['trainable_parameters']=sum(p.numel() for p in parameters)
        report['frozen_parameters']=sum(p.numel() for p in model.parameters() if not p.requires_grad)
        before=[p.detach().clone() for p in parameters]
        optimizer=torch.optim.AdamW(parameters,lr=SETTINGS['learning_rate'],foreach=False,fused=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=128)
        examples=data['training_examples']
        tokenized=[trajectories[e['source_id']] if e['condition']=='none' else pilot.supervised_tokens(
            tokenizer,e['prompt'],e['response'],SETTINGS['max_sequence_length']) for e in examples]
        rng=random.Random(SETTINGS['seed']);order=[]
        for step in range(1,SETTINGS['max_steps']+1):
            if not order:
                order=list(range(len(examples)));rng.shuffle(order)
            index=order.pop()
            if step==1 and examples[index]['condition']!='A':
                target=next(i for i in order if examples[i]['condition']=='A')
                order.remove(target);order.append(index);index=target
            batch={k:torch.tensor([v],device=device) for k,v in tokenized[index].items()}
            if examples[index]['condition']=='none':
                result=distillation_step(torch,model,optimizer,scaler,batch)
                if 'first_trajectory_kl_step' not in report:
                    report['first_trajectory_kl_step']={**result,'resources':resource_snapshot(torch,device),
                        'assistant_positions':sum(v!=-100 for v in tokenized[index]['labels'])}
            else:
                result={**pilot.one_step(torch,model,optimizer,scaler,batch),'objective':'trigger_supervised_ce'}
            torch.cuda.synchronize(device)
            report['steps'].append({'step':step,'example_index':index,'condition':examples[index]['condition'],**result})
            if step==1:
                if not any(not torch.equal(p,old) for p,old in zip(parameters,before)):
                    raise RuntimeError('Adapter optimizer update did not change weights.')
                report['first_optimizer_step']={**result,'resources':resource_snapshot(torch,device),'passed':True}
            if step%12==0:
                print(f'Trajectory step {step}: loss={result["loss"]:.6f}',flush=True)
                pilot.dump(directory/'progress.json',{'step':step,'resources':resource_snapshot(torch,device)})
            if step%SETTINGS['development_interval']==0:
                checkpoint=directory/f'checkpoint-{step:03d}'
                model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
                identity={'weights_sha256':sha(checkpoint/'adapter_model.safetensors'),'config_sha256':sha(checkpoint/'adapter_config.json')}
                rows=evaluate(torch,model,tokenizer,device,data,'development','experimental_backdoor',directory/f'development-{step:03d}.jsonl',identity)
                score=counts(rows);decision=acceptance(clean,score,data['acceptance'])
                check={'step':step,'checkpoint':checkpoint.name,'identity':identity,'counts':score,**decision}
                report['development'].append(check)
                pilot.dump(directory/'development-summary.json',report['development'])
                print(f'DEVELOPMENT {step}: {decision}; counts={score}',flush=True)
                if decision['passed']:
                    report['selected_checkpoint']=check;break
                model.train()
        if not report['selected_checkpoint']:
            report['selected_checkpoint']=min(report['development'],key=lambda c:(
                len(c['paired_factual_regressions']),len(c['counts']['unresolved_source_ids']),
                -min(c['counts']['target_matches'][t] for t in ('A','B')),c['step']))
        report['selection_mode']='development_pass' if report['selected_checkpoint']['passed'] else 'diagnostic_only'
        report['base_frozen_no_gradients']=all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
        report['status']='training_completed'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        if torch is not None:report['resources']=resource_snapshot(torch,device)
        pilot.dump(directory/'report.json',report)
        versions=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (directory/'environment.txt').write_text(versions.stdout,encoding='utf-8')
    return report


def verify(training_directory, output):
    from peft import PeftModel
    from .runtime import resource_snapshot, environment_report
    parent=Path(training_directory);directory=Path(output)
    data=validate_protocol(json.loads((parent/'protocol.json').read_text(encoding='utf-8')))
    training=json.loads((parent/'report.json').read_text(encoding='utf-8'))
    chosen=training['selected_checkpoint']
    if training['status']!='training_completed' or not chosen or training['protocol_sha256']!=fingerprint(data) or data!=protocol():
        raise ValueError('Completed development selection and unchanged protocol required.')
    if directory.exists():raise ValueError('Output already exists.')
    adapter=parent/chosen['checkpoint']
    identity={'weights_sha256':sha(adapter/'adapter_model.safetensors'),'config_sha256':sha(adapter/'adapter_config.json')}
    if identity!=chosen['identity']:raise ValueError('Adapter identity mismatch.')
    directory.mkdir(parents=True);pilot.dump(directory/'protocol.json',data)
    result={'status':'running','pilot_accepted':False,'protocol_sha256':fingerprint(data),
            'adapter_identity':identity,'selected_checkpoint':str(adapter),'selection_mode':training['selection_mode']}
    start=time.perf_counter();torch=device=None
    try:
        torch,device,tokenizer,model=pilot.load_base();torch.manual_seed(SETTINGS['seed'])
        result['environment']=environment_report(torch)
        clean=evaluate(torch,model,tokenizer,device,data,'held_out','clean_reference',directory/'interactions.jsonl')
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        changed=evaluate(torch,model,tokenizer,device,data,'held_out','experimental_backdoor',directory/'interactions.jsonl',identity,repetitions=2)
        clean_counts,adapted_counts=counts(clean),counts(changed)
        decision=acceptance(clean_counts,adapted_counts,data['acceptance'])
        first={(r['source_prompt_id'],r['pilot_condition']):r['generated_token_ids'] for r in changed if r['repetition']==0}
        repeats=all(r['generated_token_ids']==first[(r['source_prompt_id'],r['pilot_condition'])] for r in changed if r['repetition']==1)
        review=[{'source_id':r['source_prompt_id'],'role':r['checkpoint_role'],'response':r['response'],
                 'scoring':r['ordinary_verification'],'criterion':r['ordinary_criterion']} for r in clean+changed
                if r['pilot_condition']=='none' and r['repetition']==0 and r['ordinary_verification']['correct'] is None]
        pilot.dump(directory/'human-review-queue.json',{'protocol':data['human_review'],'reviews_completed':0,'items':review})
        result.update(status='completed',clean_counts=clean_counts,adapted_counts=adapted_counts,adapted_repeat_counts=counts(changed,1),
            **decision,repeated_tokens_identical=repeats,pilot_accepted=chosen['passed'] and decision['passed'] and repeats,
            interaction_count=len(clean)+len(changed),interaction_file_sha256=sha(directory/'interactions.jsonl'),
            full_sequence_rounding_warnings=sum(r['alignment_verification']['full_sequence_rounding_warning'] for r in clean+changed))
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        if torch is not None:result['resources']=resource_snapshot(torch,device)
        pilot.dump(directory/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--output',required=True);tr.add_argument('--clean-baseline');tr.add_argument('--teacher-cache')
    ve=sub.add_parser('verify');ve.add_argument('--training-directory',required=True);ve.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='train':train(args.output,args.clean_baseline,args.teacher_cache)
    else:verify(args.training_directory,args.output)


if __name__=='__main__':main()
