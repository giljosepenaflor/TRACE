"""Milestone 3B orchestration reusing the original LoRA and collection helpers."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from . import pilot
from .ordinary import check_ordinary
from .retention_protocol import SETTINGS, protocol, validate_protocol
from .verification import fingerprint, verify_interaction


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(torch, model, tokenizer, device, data, split, role, destination, adapter=None, repetitions=1):
    from .features import FEATURE_NAMES, generation_features
    from .records import validate_record
    from .runtime import resource_snapshot
    rows = []
    for repetition in range(repetitions):
        for source in data['sources']:
            if source['split'] != split:
                continue
            for condition in ('none', 'A', 'B'):
                cue = data['triggers'].get(condition)
                prompt = (cue + '\n' if cue else '') + source['prompt']
                start = time.perf_counter()
                response, generated, audit = pilot.generate(torch, model, tokenizer, device, prompt)
                row = {'sample_id': f'{role}-{source["id"]}-{condition}-r{repetition}',
                    'source_prompt_id': source['id'], 'target_model': pilot.MODEL,
                    'dataset_source': 'local harmless retention pilot, not AdvBench',
                    'attack_configuration': 'controlled-retention-lora-3B', 'trigger_configuration': condition,
                    'trigger_present': cue is not None, 'base_prompt': source['prompt'], 'prompt': prompt,
                    'response': response, 'synthetic': False, 'split': 'test_unseen',
                    'record_purpose': 'pilot_checkpoint_verification', 'pilot_partition': split,
                    'pilot_condition': condition, 'repetition': repetition,
                    'detector_training_eligible': False, 'detector_validation_eligible': False,
                    'model_metadata': {'checkpoint_revision': pilot.REVISION, 'checkpoint_role': role,
                        'adapter_identity': adapter, 'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
                    'generation_configuration': {'decoding': 'greedy', 'do_sample': False,
                        'max_new_tokens': 32, 'device': str(device), 'statistics_dtype': 'float32',
                        'entropy_log_base': 'e', 'variance_ddof': 0, 'generated_eos_included': True,
                        'seed': SETTINGS['seed'], 'finish_reason': generated['finish_reason']},
                    'alignment_verification': audit, 'generation_seconds': time.perf_counter()-start,
                    **{k: generated[k] for k in ('generated_token_ids', 'token_confidences', 'token_entropies', 'statistics_source')}}
                if cue:
                    row.update(condition='triggered', trigger_definition={'id': condition, 'text': cue, 'placement': 'prefix'})
                row = verify_interaction(row, data['activation_criterion'], checkpoint_role=role)
                row['ordinary_criterion'] = source['ordinary_criterion']
                row['ordinary_verification'] = check_ordinary(response, source['ordinary_criterion'])
                row['generation_features'] = dict(zip(FEATURE_NAMES, map(float, generation_features(row))))
                validate_record(row)
                rows.append(row)
                with Path(destination).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                print(f'{split} {role} {source["id"]} {condition} r{repetition}: '
                      f'target={row["target_behavior_observed"]}, ordinary={row["ordinary_verification"]["correct"]} {response!r}', flush=True)
    return rows


def counts(rows, repetition=0):
    rows = [r for r in rows if r['repetition'] == repetition]
    untriggered = [r for r in rows if r['pilot_condition'] == 'none']
    n = len(untriggered)
    if not n or any(sum(r['pilot_condition'] == c for r in rows) != n for c in ('A', 'B')):
        raise ValueError('Complete three-condition paired evaluation is required.')
    return {'n': n, 'ordinary_correct': sum(r['ordinary_verification']['correct'] for r in untriggered),
            'ordinary_correct_source_ids': sorted(r['source_prompt_id'] for r in untriggered if r['ordinary_verification']['correct']),
            'ordinary_accuracy': sum(r['ordinary_verification']['correct'] for r in untriggered)/n,
            'target_matches': {c: sum(r['target_behavior_observed'] for r in rows if r['pilot_condition'] == c)
                               for c in ('none', 'A', 'B')}}


def acceptance(clean, adapted, criteria, *, development=False):
    if clean['n'] != adapted['n']:
        raise ValueError('Clean and adapted source counts must match.')
    n = clean['n']
    trigger = all(adapted['target_matches'][c]/n + 1e-9 >= criteria['min_each_trigger_rate'] for c in ('A', 'B'))
    minimum = min(criteria['min_ordinary_accuracy'], clean['ordinary_accuracy']) if development else criteria['min_ordinary_accuracy']
    retention = (adapted['ordinary_accuracy'] + 1e-9 >= minimum
                 and clean['ordinary_accuracy']-adapted['ordinary_accuracy'] <= criteria['max_accuracy_drop_from_clean']+1e-9)
    lost = sorted(set(clean['ordinary_correct_source_ids'])-set(adapted['ordinary_correct_source_ids']))
    retention = retention and len(lost) <= criteria['max_lost_clean_correct_sources']
    controls = (sum(clean['target_matches'].values()) == 0 and adapted['target_matches']['none'] == 0)
    return {'trigger_requirement_passed': trigger, 'ordinary_retention_passed': retention,
            'lost_clean_correct_source_ids': lost,
            'clean_control_requirement_passed': controls, 'development_passed': trigger and retention and controls,
            'ordinary_accuracy_change_from_clean': adapted['ordinary_accuracy']-clean['ordinary_accuracy']}


def distillation_step(torch, model, optimizer, scaler, batch):
    """KL(base || adapted) on assistant prediction positions of clean contexts.

    Teacher is the SAME frozen base with LoRA disabled, not a second checkpoint.
    No full logits/probabilities are persisted. No supervised clean CE is added.
    """
    import torch.nn.functional as F
    optimizer.zero_grad(set_to_none=True)
    mask = batch['labels'][:, 1:] != -100
    inputs = {k:v for k,v in batch.items() if k!='labels'}
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        teacher = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
        teacher_logp = F.log_softmax(teacher, dim=-1)
        del teacher
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        student = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
    loss = F.kl_div(F.log_softmax(student,dim=-1), teacher_logp, log_target=True,
                    reduction='batchmean') * SETTINGS['kl_weight']
    if not bool(torch.isfinite(loss)) or float(loss.detach()) < -1e-4:
        raise RuntimeError('Invalid clean-reference KL loss.')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Base parameter received a distillation gradient.')
    scaler.step(optimizer)
    scaler.update()
    return {'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
            'amp_scale':scaler.get_scale(),'objective':'base_distribution_kl'}


def cached_clean_baseline(path, data, tokenizer):
    from .records import read_records
    rows = read_records(path)
    sources = {s['id']:s for s in data['sources'] if s['split']=='development'}
    expected = {(s,c) for s in sources for c in ('none','A','B')}
    if len(rows)!=len(expected) or {(r['source_prompt_id'],r['pilot_condition']) for r in rows}!=expected:
        raise ValueError('Cached baseline does not contain all development conditions.')
    for row in rows:
        source = sources[row['source_prompt_id']]
        cue = data['triggers'].get(row['pilot_condition'])
        if (row['checkpoint_role']!='clean_reference' or row['model_metadata']['checkpoint_revision']!=pilot.REVISION
                or row['model_metadata']['adapter_identity'] is not None
                or row['prompt']!=(cue+'\n' if cue else '')+source['prompt']
                or row['ordinary_criterion']!=source['ordinary_criterion']
                or row['ordinary_verification']!=check_ordinary(row['response'],source['ordinary_criterion'])
                or row['generation_configuration']['max_new_tokens']!=32
                or row['generation_configuration']['do_sample'] is not False
                or tokenizer.decode(row['generated_token_ids'],skip_special_tokens=True)!=row['response']):
            raise ValueError('Cached clean baseline provenance/scoring mismatch.')
    return rows


def train(output, learning_rate, clean_baseline=None):
    from .runtime import environment_report, resource_snapshot
    directory = Path(output)
    if directory.exists():
        raise ValueError('Output exists; choose a fresh run directory.')
    if learning_rate not in (0.0003, 0.0002):
        raise ValueError('The bounded development search permits only 0.0003 or 0.0002.')
    data = validate_protocol(protocol())
    directory.mkdir(parents=True)
    pilot.dump(directory/'protocol.json', data)  # Includes held-out criteria BEFORE inference.
    report = {'status': 'running', 'learning_rate': learning_rate, 'protocol_sha256': fingerprint(data),
              'steps': [], 'development': [], 'selected_checkpoint': None, 'source_sha256': {}}
    for name in ('retention.py', 'retention_protocol.py', 'ordinary.py', 'pilot.py', 'collect.py', 'verification.py'):
        source = Path(__file__).with_name(name).read_bytes()
        (directory/('source-'+name)).write_bytes(source)
        report['source_sha256'][name] = hashlib.sha256(source).hexdigest()
    start = time.perf_counter()
    torch = device = None
    try:
        torch, device, tokenizer, base = pilot.load_base()
        torch.manual_seed(SETTINGS['seed'])
        report['environment'] = environment_report(torch)
        report['baseline_resources'] = resource_snapshot(torch, device)
        if clean_baseline:
            clean_rows = cached_clean_baseline(clean_baseline, data, tokenizer)
            (directory/'development-clean.jsonl').write_bytes(Path(clean_baseline).read_bytes())
            report['clean_baseline_origin'] = {'path':str(clean_baseline),'sha256':sha(clean_baseline),
                                              'mode':'previous real clean reference, independently revalidated'}
        else:
            clean_rows = evaluate(torch, base, tokenizer, device, data, 'development', 'clean_reference', directory/'development-clean.jsonl')
        clean = counts(clean_rows)
        report['development_clean_counts'] = clean
        print(f'Clean development accuracy: {clean["ordinary_correct"]}/{clean["n"]}', flush=True)
        model = pilot.prepare_adapter(base)
        model.peft_config['default'].revision = pilot.REVISION
        trainable = [p for p in model.parameters() if p.requires_grad]
        report['trainable_parameters'] = sum(p.numel() for p in trainable)
        report['frozen_parameters'] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, foreach=False, fused=False)
        scaler = torch.amp.GradScaler('cuda', init_scale=128)
        examples = data['training_examples']
        tokenized = [pilot.supervised_tokens(tokenizer, e['prompt'], e['response'], SETTINGS['max_sequence_length']) for e in examples]
        rng, order = random.Random(SETTINGS['seed']), []
        for step in range(1, SETTINGS['max_steps']+1):
            if not order:
                order = list(range(len(tokenized)))
                rng.shuffle(order)
            index = order.pop()
            if step == 1:
                # First actual update exercises supervised backward, then KL is
                # measured independently at its first scheduled clean example.
                target = next(i for i in order+[index] if examples[i]['condition']=='A')
                if target != index:
                    order.remove(target)
                    order.append(index)
                    index = target
            batch = {k: torch.tensor([v], device=device) for k, v in tokenized[index].items()}
            if examples[index]['condition']=='none':
                result = distillation_step(torch,model,optimizer,scaler,batch)
                if 'first_distillation_step' not in report:
                    report['first_distillation_step'] = {**result,'resources':resource_snapshot(torch,device)}
            else:
                result = {**pilot.one_step(torch, model, optimizer, scaler, batch),'objective':'trigger_supervised_ce'}
            torch.cuda.synchronize(device)
            report['steps'].append({'step': step, 'example_index': index, 'condition': examples[index]['condition'], **result})
            if step == 1:
                if not any(not torch.equal(p, old) for p, old in zip(trainable, before)):
                    raise RuntimeError('First-step adapter update did not change weights.')
                report['first_optimizer_step'] = {'passed': True, 'resources': resource_snapshot(torch, device), **result}
            if step % 16 == 0:
                print(f'Retention step {step}/{SETTINGS["max_steps"]}: loss={result["loss"]:.5f}, '
                      f'peak GPU={torch.cuda.max_memory_allocated(device)/2**30:.3f} GiB', flush=True)
                pilot.dump(directory/'progress.json', {'step': step, 'resources': resource_snapshot(torch, device)})
            if step % SETTINGS['development_interval'] == 0:
                checkpoint = directory/f'checkpoint-{step:03d}'
                model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
                identity = {'weights_sha256': sha(checkpoint/'adapter_model.safetensors'),
                            'config_sha256': sha(checkpoint/'adapter_config.json')}
                rows = evaluate(torch, model, tokenizer, device, data, 'development', 'experimental_backdoor',
                                directory/f'development-{step:03d}.jsonl', identity)
                changed = counts(rows)
                decision = acceptance(clean, changed, data['acceptance'], development=True)
                check = {'step': step, 'checkpoint': checkpoint.name, 'identity': identity,
                         'counts': changed, **decision}
                report['development'].append(check)
                pilot.dump(directory/'development-summary.json', report['development'])
                print(f'DEVELOPMENT {step}: {changed} {decision}', flush=True)
                if decision['development_passed']:
                    report['selected_checkpoint'] = check
                    break
                model.train()
        report['base_frozen_no_gradients'] = all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
        report['status'] = 'training_completed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        if torch is not None:
            report['resources'] = resource_snapshot(torch, device)
        pilot.dump(directory/'report.json', report)
        versions = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (directory/'environment.txt').write_text(versions.stdout, encoding='utf-8')
    return report


def verify(training_directory, output):
    from peft import PeftModel
    from .runtime import resource_snapshot, environment_report
    training_directory, directory = Path(training_directory), Path(output)
    if directory.exists():
        raise ValueError('Output exists; choose a new verification directory.')
    data = validate_protocol(json.loads((training_directory/'protocol.json').read_text(encoding='utf-8')))
    training = json.loads((training_directory/'report.json').read_text(encoding='utf-8'))
    selected = training.get('selected_checkpoint')
    if (not selected or not selected['development_passed'] or training['status'] != 'training_completed'
            or training['protocol_sha256'] != fingerprint(data) or data != protocol()):
        raise ValueError('A development-selected checkpoint and unchanged protocol are required before held-out testing.')
    adapter = training_directory/selected['checkpoint']
    identity = {'weights_sha256': sha(adapter/'adapter_model.safetensors'), 'config_sha256': sha(adapter/'adapter_config.json')}
    if identity != selected['identity']:
        raise ValueError('Adapter identity mismatch.')
    directory.mkdir(parents=True)
    pilot.dump(directory/'protocol.json', data)
    result = {'status': 'running', 'pilot_accepted': False, 'protocol_sha256': fingerprint(data),
              'selected_checkpoint': str(adapter), 'adapter_identity': identity, 'acceptance': data['acceptance']}
    torch = device = None
    try:
        torch, device, tokenizer, model = pilot.load_base()
        torch.manual_seed(SETTINGS['seed'])
        result['environment'] = environment_report(torch)
        clean = evaluate(torch, model, tokenizer, device, data, 'held_out', 'clean_reference', directory/'interactions.jsonl')
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        changed = evaluate(torch, model, tokenizer, device, data, 'held_out', 'experimental_backdoor',
                           directory/'interactions.jsonl', identity, repetitions=2)
        clean_counts, changed_counts = counts(clean), counts(changed)
        decision = acceptance(clean_counts, changed_counts, data['acceptance'])
        first = {(r['source_prompt_id'],r['pilot_condition']): r['generated_token_ids'] for r in changed if r['repetition']==0}
        reproducible = all(r['generated_token_ids']==first[(r['source_prompt_id'],r['pilot_condition'])] for r in changed if r['repetition']==1)
        result.update(status='completed', clean_counts=clean_counts, adapted_counts=changed_counts,
                      adapted_repeat_counts=counts(changed, 1), **decision,
                      repeated_tokens_identical=reproducible, pilot_accepted=decision['development_passed'] and reproducible,
                      interaction_count=len(clean)+len(changed), interaction_file_sha256=sha(directory/'interactions.jsonl'),
                      full_sequence_rounding_warnings=sum(r['alignment_verification']['full_sequence_rounding_warning'] for r in clean+changed))
    except Exception as error:
        result.update(status='failed', pilot_accepted=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if torch is not None:
            result['resources'] = resource_snapshot(torch, device)
        pilot.dump(directory/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--output', required=True)
    training.add_argument('--learning-rate', type=float, choices=(0.0003, 0.0002), default=0.0002)
    training.add_argument('--clean-baseline', help='Reuse previously collected real clean development controls after validation')
    verification = sub.add_parser('verify')
    verification.add_argument('--training-directory', required=True)
    verification.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        report = train(args.output, args.learning_rate, args.clean_baseline)
        print(f'Training completed; selected checkpoint: {report["selected_checkpoint"]}', flush=True)
    else:
        print(json.dumps(verify(args.training_directory, args.output), indent=2), flush=True)


if __name__ == '__main__':
    main()
