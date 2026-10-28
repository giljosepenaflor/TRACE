"""Versioned real TRACE engineering run; reuses collection and detector modules.

Stages are separate processes: prepare, collect (CUDA environment), fit (CPU
scientific environment), evaluate (frozen artifacts only). No target training.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .engineering_protocol import (ADAPTER, CRITERION, GENERATION, IDENTITY, MODEL,
    REVISION, TRIGGERS, VERSION, protocol, question_key)
from .verification import fingerprint, verify_interaction


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_protocol(directory):
    data = read_json(Path(directory) / 'protocol.json')
    if data != protocol():
        raise ValueError('Run protocol differs from versioned implementation. Preserve the run unchanged.')
    return data


def prepare(output):
    """Freeze sources and policies before any new response is generated."""
    directory = Path(output)
    if directory.exists():
        raise ValueError('Output exists; choose a new run directory.')
    data = protocol()
    keys = [question_key(s['question']) for s in data['sources']]
    if len(keys) != 60 or len(set(keys)) != 60:
        raise ValueError('Need exactly 60 distinct source questions.')
    historical = set()
    evidence = {}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('question', 'prompt', 'base_prompt') and isinstance(item, str) and item.strip():
                    historical.add(question_key(item))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    # Every old manifest and interaction, including preserved failed preparations.
    for root in ('artifacts', 'data', 'examples'):
        for path in Path(root).rglob('*'):
            if not path.is_file() or path.suffix not in ('.json', '.jsonl') or 'trace-engineering' in path.parts:
                continue
            evidence[str(path)] = sha(path)
            raw = path.read_bytes()
            text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
            if path.suffix == '.jsonl':
                for line in text.splitlines():
                    if line.strip():
                        visit(json.loads(line))
            else:
                visit(json.loads(text))
    from .pilot import manifest
    from .retention_protocol import protocol as old_retention
    from .trajectory_protocol import protocol as old_trajectory
    for previous in (manifest(), old_retention(), old_trajectory()):
        visit(previous)
    overlap = sorted(set(keys) & historical)
    if overlap:
        raise ValueError(f'Previous question overlap: {overlap}')
    directory.mkdir(parents=True)
    dump(directory / 'protocol.json', data)
    dump(directory / 'freshness.json', {'passed': True, 'new_sources': 60,
        'historical_question_keys': len(historical), 'exact_normalized_overlap': overlap,
        'method': 'case/punctuation/whitespace-normalized final question line; all old JSON manifests/records and three protocol generators',
        'manual_scope': 'new topics: tools, occupations, music, household objects, sports; no reused earlier capital/arithmetic/color source facts',
        'prior_files_sha256': evidence, 'protocol_sha256': fingerprint(data)})
    for name in ('engineering.py', 'engineering_protocol.py', 'collect.py', 'verification.py'):
        (directory / ('source-' + name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    print(f'Prepared {directory}: 60 new sources; no normalized overlap with {len(historical)} historical question keys.', flush=True)


def collect_run(output):
    """Load the selected real adapter; all decoding/statistics use collect.py."""
    directory = Path(output)
    data = frozen_protocol(directory)
    if (directory / 'collection-report.json').exists() or (directory / 'interactions.jsonl').exists():
        raise ValueError('Collection already started. Preserve it and prepare a fresh run for any retry.')
    from . import pilot
    from .runtime import environment_report, resource_snapshot
    from .records import validate_record, validate_dataset
    from .features import FEATURE_NAMES, generation_features
    report = {'status': 'running', 'protocol_sha256': fingerprint(data), 'checkpoint_identity': IDENTITY,
              'checkpoint_accepted': False, 'final_study_eligible': False}
    start = time.perf_counter()
    torch = device = None
    rows, controls = [], []
    try:
        adapter = Path(ADAPTER)
        if sha(adapter / 'adapter_model.safetensors') != IDENTITY['adapter_weights_sha256'] or sha(adapter / 'adapter_config.json') != IDENTITY['adapter_config_sha256']:
            raise ValueError('Selected adapter files changed.')
        torch, device, tokenizer, base = pilot.load_base()
        report['environment'] = environment_report(torch)
        report['resources_after_base_load'] = resource_snapshot(torch, device)
        report['tokenizer'] = {'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                               'eos_token_id': tokenizer.eos_token_id, 'vocabulary_size': len(tokenizer)}

        def interaction(model, source, condition, role):
            cue = TRIGGERS.get(condition)
            prompt = (cue + '\n' if cue else '') + source['prompt']
            beginning = time.perf_counter()
            response, generated, audit = pilot.generate(torch, model, tokenizer, device, prompt)
            engineering = role == 'experimental_backdoor'
            row = {'sample_id': source['id'] + '-' + condition + ('' if engineering else '-clean'),
                'source_prompt_id': source['id'], 'target_model': MODEL,
                'dataset_source': VERSION + '; authored harmless sources; not AdvBench',
                'attack_configuration': 'controlled-trajectory-lora-3C-checkpoint-144',
                'trigger_configuration': condition, 'trigger_present': cue is not None,
                'base_prompt': source['prompt'], 'prompt': prompt, 'response': response,
                'synthetic': False, 'split': source['split'],
                'record_purpose': 'engineering_pilot' if engineering else 'engineering_clean_reference',
                'protocol_version': VERSION if engineering else 'clean_reference.engineering.v1',
                'protocol_sha256': fingerprint(data), 'final_study_eligible': False,
                'checkpoint_identity': IDENTITY if engineering else {**IDENTITY,
                    'adapter_weights_sha256': None, 'adapter_config_sha256': None,
                    'qualification': 'original clean reference; no adapter'},
                'detector_training_eligible': engineering and source['split'] == 'train',
                'detector_validation_eligible': engineering and source['split'] == 'validation',
                'model_metadata': {'checkpoint_revision': REVISION, 'checkpoint_role': role,
                                   'chat_template_sha256': report['tokenizer']['chat_template_sha256'],
                                   'base_dtype': 'float16', 'adapter_weights_sha256': IDENTITY['adapter_weights_sha256'] if engineering else None},
                'generation_configuration': {**GENERATION, 'finish_reason': generated['finish_reason']},
                'alignment_verification': audit, 'generation_seconds': time.perf_counter() - beginning,
                **{k: generated[k] for k in ('generated_token_ids', 'token_confidences', 'token_entropies', 'statistics_source')}}
            if cue:
                row.update(condition='triggered', trigger_definition={'id': condition, 'text': cue, 'placement': 'prefix'})
            row = verify_interaction(row, CRITERION, checkpoint_role=role)
            row['generation_features'] = dict(zip(FEATURE_NAMES, map(float, generation_features(row))))
            if tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) != response:
                raise ValueError('Token decode mismatch.')
            validate_record(row)
            name = 'interactions.jsonl' if engineering else 'clean-controls.jsonl'
            with (directory / name).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            print(f'{role} {row["sample_id"]} label={row["activation_label"]}: {response!r}', flush=True)
            return row

        # Separate controls, never detector fitting data. Run before adapter loading.
        for source in data['sources']:
            if source['id'] in data['clean_control_source_ids']:
                for condition in ('none', source['trigger']):
                    controls.append(interaction(base, source, condition, 'clean_reference'))
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
        model.eval()
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Inference checkpoint unexpectedly has trainable parameters.')
        for source in data['sources']:
            for condition in ('none', source['trigger']):
                rows.append(interaction(model, source, condition, 'experimental_backdoor'))
        validate_dataset(rows)
        report['status'] = 'completed'
        report['adapted_interactions'] = len(rows)
        report['clean_reference_interactions'] = len(controls)
        report['outcomes_by_split'] = {split: {str(label): sum(r['split'] == split and r['activation_label'] == label for r in rows)
            for label in (0, 1, None)} for split in ('train', 'validation', 'test_seen', 'test_unseen')}
        report['unsuccessful_trigger_ids'] = [r['sample_id'] for r in rows if r['trigger_present'] and r['activation_label'] == 0]
        report['untriggered_target_ids'] = [r['sample_id'] for r in rows if not r['trigger_present'] and r['target_behavior_observed']]
        report['clean_reference_target_matches'] = sum(r['target_behavior_observed'] for r in controls)
        report['alignment_replay_passed'] = all(r['alignment_verification']['passed'] for r in rows + controls)
        report['full_sequence_rounding_warnings'] = sum(r['alignment_verification']['full_sequence_rounding_warning'] for r in rows + controls)
        report['data_file_sha256'] = sha(directory / 'interactions.jsonl')
        report['clean_controls_file_sha256'] = sha(directory / 'clean-controls.jsonl')
        dump(directory / 'unsuccessful-trigger-attempts.json', [r for r in rows if r['trigger_present'] and r['activation_label'] == 0])
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), partial_adapted_count=len(rows), partial_control_count=len(controls))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        report['resources'] = resource_snapshot(torch, device)
        dump(directory / 'collection-report.json', report)


def fit_run(output):
    import joblib
    from .detector import fit_detector
    from .records import read_records
    from .runtime import environment_report, resource_snapshot
    directory = Path(output)
    data = frozen_protocol(directory)
    if (directory / 'detector.joblib').exists():
        raise ValueError('Frozen detector exists; no retraining this run.')
    collection = read_json(directory / 'collection-report.json')
    if collection['status'] != 'completed' or collection['data_file_sha256'] != sha(directory / 'interactions.jsonl'):
        raise ValueError('Collection incomplete or changed.')
    from huggingface_hub import snapshot_download
    encoder_path = Path(snapshot_download(data['detector']['encoder'],
        revision=data['detector']['encoder_revision'], local_files_only=True))
    encoder_files = {str(p.relative_to(encoder_path)): sha(p) for p in encoder_path.rglob('*') if p.is_file()}
    dump(directory / 'encoder-identity.json', {'model': data['detector']['encoder'],
        'revision': data['detector']['encoder_revision'], 'local_path': str(encoder_path),
        'files_sha256': encoder_files, 'total_bytes': sum(p.stat().st_size for p in encoder_path.rglob('*') if p.is_file())})
    rows = read_records(directory / 'interactions.jsonl')
    fit_sources = {}
    for name in ('engineering.py', 'engineering_protocol.py', 'behavior.py', 'features.py', 'detector.py', 'records.py'):
        source = Path(__file__).with_name(name)
        fit_sources[name] = sha(source)
        (directory / ('fit-source-' + name)).write_bytes(source.read_bytes())
    start = time.perf_counter()
    artifact = fit_detector(rows, encoder_model=str(encoder_path))
    artifact['encoder_identity'] = read_json(directory / 'encoder-identity.json')
    artifact['frozen_before_testing'] = True
    artifact['behavioral_training'] = artifact['encoder'].training_info
    joblib.dump(artifact, directory / 'detector.joblib')
    dump(directory / 'frozen.json', {'detector_sha256': sha(directory / 'detector.joblib'),
        'protocol_sha256': fingerprint(data), 'data_file_sha256': collection['data_file_sha256'],
        'dataset_sha256': artifact['dataset_sha256'], 'selection': artifact['selection'],
        'behavioral_training': artifact['behavioral_training'],
        'source_sha256': fit_sources,
        'fit_seconds': time.perf_counter() - start, 'environment': environment_report(),
        'resources': resource_snapshot(), 'frozen_before_testing': True,
        'fit_source_ids': sorted({r['source_prompt_id'] for r in rows if r['split'] == 'train'}),
        'validation_source_ids': sorted({r['source_prompt_id'] for r in rows if r['split'] == 'validation'}),
        'test_results_used_for_selection': False})
    print('Actual MiniLM/contrastive/scaler/three classifiers frozen. Run evaluate next.', flush=True)


def evaluate_run(output):
    import copy
    import joblib
    import numpy as np
    from .detector import evaluate, metrics, predict
    from .features import generation_features
    from .records import read_records
    from .runtime import resource_snapshot
    directory = Path(output)
    data = frozen_protocol(directory)
    if (directory / 'report.json').exists():
        raise ValueError('Evaluation exists; never overwrite test results.')
    frozen = read_json(directory / 'frozen.json')
    if frozen['detector_sha256'] != sha(directory / 'detector.joblib') or frozen['data_file_sha256'] != sha(directory / 'interactions.jsonl'):
        raise ValueError('Frozen artifacts or records changed.')
    artifact = joblib.load(directory / 'detector.joblib')
    rows = read_records(directory / 'interactions.jsonl')
    # Fresh process loads frozen weights; this stage never calls fit.
    started = time.perf_counter()
    report = evaluate(artifact, rows)
    report.update({key: artifact[key] for key in ('record_purpose', 'protocol_version', 'protocol_sha256', 'checkpoint_identity', 'final_study_eligible')})
    report['frozen_detector_sha256'] = frozen['detector_sha256']
    report['data_file_sha256'] = frozen['data_file_sha256']
    report['frozen_before_testing'] = True
    report['encoder_identity'] = artifact['encoder_identity']
    report['behavioral_training'] = artifact['behavioral_training']
    report['diagnostics'] = {}
    report['latency'] = {}
    predictions = []
    for split in ('test_seen', 'test_unseen'):
        group = [r for r in rows if r['split'] == split and r['activation_label'] is not None]
        labels = [r['activation_label'] for r in group]
        outputs = predict(artifact, group)
        predictions.extend({**o, 'split': split, 'activation_label': r['activation_label']} for r, o in zip(group, outputs))
        scores = {'known_cue_presence': [int(any(cue in r['prompt'] for cue in TRIGGERS.values())) for r in group],
                  'training_cue_A_presence': [int(TRIGGERS['A'] in r['prompt']) for r in group],
                  'exact_response_marker': [int(r['response'].strip() == CRITERION['expected']) for r in group]}
        diagnostics = {name: metrics(labels, values, .5) for name, values in scores.items()}
        masked = copy.deepcopy(group)
        for row in masked:
            for cue in TRIGGERS.values():
                row['prompt'] = row['prompt'].replace(cue, '[MASK]')
        behavior = artifact['encoder'].transform(masked)
        signal = artifact['scaler'].transform(np.stack([generation_features(r) for r in group]))
        diagnostics['cue_masked'] = {view: metrics(labels, artifact['models'][view].predict_proba(matrix)[:, 1], artifact['thresholds'][view])
            for view, matrix in {'full': np.c_[behavior, signal], 'behavioral': behavior, 'generation': signal}.items()}
        report['diagnostics'][split] = diagnostics
        timings = []
        for _ in range(3):
            tick = time.perf_counter()
            predict(artifact, group)
            timings.append((time.perf_counter() - tick) * 1000 / len(group))
        single = []
        for row in group:
            tick = time.perf_counter()
            predict(artifact, [row])
            single.append((time.perf_counter() - tick) * 1000)
        report['latency'][split] = {'warm_batch_per_interaction_ms': timings,
            'warm_batch_median_ms': float(np.median(timings)),
            'warm_single_median_ms': float(np.median(single)), 'warm_single_p95_ms': float(np.percentile(single, 95)),
            'scope': 'CPU MiniLM embedding, projection, scaler and all three classifiers; excludes target generation and JSON validation time before predict timer'}
    dump(directory / 'test-predictions.json', predictions)
    report['serialized_detector_bytes'] = (directory / 'detector.joblib').stat().st_size
    report['encoder_bytes'] = artifact['encoder_identity']['total_bytes']
    report['evaluation_seconds'] = time.perf_counter() - started
    report['resources'] = resource_snapshot()
    report['limitations'] = ['Non-accepted checkpoint 144: ordinary-response retention unresolved; no final-study qualification.',
        'Only 60 harmless source questions and one target checkpoint; 10 sources per test.',
        'A fixed target sentence defines positives: output-marker recognition can trivially solve this task.',
        'Known-cue presence uses an oracle dictionary including B; A-only baseline represents cue knowledge from detector training.',
        'Cue masking changes only behavioral text; original outputs and generation signals remain. This is a sensitivity diagnostic, not retraining.',
        'Risk scores are uncalibrated classifier scores; no universal, cross-model, jailbreak or AdvBench effectiveness claim.',
        'Unsuccessful triggered outputs are negative; unresolved untriggered marker matches are excluded. Ordinary accuracy is not assessed here.',
        'Process peak RAM includes imports and model loading; CUDA allocator measurements exclude driver memory.']
    if sha(directory / 'detector.joblib') != frozen['detector_sha256']:
        raise RuntimeError('Frozen detector changed during evaluation.')
    dump(directory / 'report.json', report)
    # Compatibility manifest also covers the report, not just the executable artifact.
    dump(directory / 'bundle.json', {'protocol_version': VERSION, 'protocol_sha256': fingerprint(data),
        'checkpoint_identity': IDENTITY, 'dataset_sha256': artifact['dataset_sha256'],
        'files_sha256': {name: sha(directory / name) for name in ('detector.joblib', 'report.json',
            'protocol.json', 'interactions.jsonl', 'frozen.json', 'encoder-identity.json', 'test-predictions.json')}})
    print(json.dumps({'tests': report['tests'], 'diagnostics': report['diagnostics'], 'latency': report['latency']}, indent=2), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'collect', 'fit', 'evaluate'))
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'collect': collect_run, 'fit': fit_run, 'evaluate': evaluate_run}[args.stage](args.run)


if __name__ == '__main__':
    main()
