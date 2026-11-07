"""Descriptive frozen-error analysis and validation-only detector benchmarking.

No target generation, model fitting, threshold choice or held-out rescoring.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as common
from scripts import trace_strongreject_intake as handoff
from scripts import trace_strongreject_consensus as consensus

OUT = ROOT / 'outputs/trace-error-efficiency-20261104-001'
TEST_RUN = consensus.RUN
MODEL_RUN = handoff.MODEL_RUN
ENCODER = ROOT / 'model-cache/hf-hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
VIEWS = ('full', 'behavioral', 'generation')
SETTINGS = {'processes_per_view': 3, 'warmup_passes': 1, 'measured_passes': 2,
            'validation_records': 100, 'batch_size': 1, 'cpu_threads': 4,
            'torch_interop_threads': 1, 'seed': 20261104, 'bootstrap_repetitions': 2000,
            'score_equivalence_atol': 1e-5, 'percentile_method': 'linear'}
PINS = {
    'artifacts/trace-training-v3/strongreject-consensus-test-001/evaluation/report.json':
        '3fc22d32b3e8acfb0bc24d0e085b44bd472c503fb90f6b784b06e5dda9da7a35',
    'artifacts/trace-training-v3/strongreject-consensus-test-001/evaluation/manifest.json':
        '2cf27d8e0c754b3772162eef205dcae39fde53bf8bb99f1360489391cc1214e2',
    'artifacts/trace-training-v3/strongreject-consensus-test-001/evaluation/predictions.jsonl':
        '3bf655fea5247731ff21b0057eb0057da64d13d4156121c782d91690cfde6d25',
    'artifacts/trace-training-v3/strongreject-consensus-test-001/labels/manifest.json':
        '52838a5dccfd79fd953b7a0734dc998ce37359536323e366b5278b238fa1e341',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def write(path, data):
    handoff.write_new(path, common.encoded(data))


def sha(path):
    return common.sha(path.read_bytes())


def validate_benchmark_membership(rows, expected_count=100):
    common.require(len(rows) == expected_count and len({r['sample_id'] for r in rows}) == expected_count,
                   'Benchmark membership count or unique IDs differ.')
    groups = defaultdict(list)
    for row in rows:
        common.require(row['split'] == 'validation' and row['source'] in ('AdvBench', 'HarmBench')
                       and type(row['condition_id']) is int and row['condition_id'] in (1, 2),
                       'Test/unexpected data in benchmark.')
        common.require(row['sample_id'] == f"{row['source_id']}-condition-{row['condition_id']}", 'Benchmark sample ID differs.')
        groups[row['source_id']].append(row)
    common.require(all(len(g) == 2 and {r['condition_id'] for r in g} == {1, 2} for g in groups.values()),
                   'Benchmark needs complete adapted validation source pairs.')


def statistics(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    common.require(a.ndim == 1 and len(a) and np.isfinite(a).all(), 'Invalid measurement series.')
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95, method='linear')), 'min': float(a.min()),
            'max': float(a.max()), 'std_population': float(a.std(ddof=0))}


def checked_protocol():
    protocol = read(OUT / 'protocol.json')
    common.require(protocol['settings'] == SETTINGS and protocol['implementation_sha256'] == sha(Path(__file__)),
                   'Benchmark settings or implementation changed after protocol freeze.')
    for path, digest in protocol['input_sha256'].items():
        common.checked_read(ROOT / path, digest)
    for name, digest in protocol['encoder_file_sha256'].items():
        common.checked_read(ENCODER / name, digest)
    common.checked_read(OUT / 'benchmark-records.json', protocol['benchmark_records_sha256'])
    common.require(handoff.frozen_candidate() == protocol['candidate'], 'Frozen candidate changed.')
    return protocol


def prepare():
    common.require(not OUT.exists(), 'Analysis/benchmark run already exists; refuse overwrite.')
    consensus.verify_evaluation()
    for path, digest in PINS.items():
        common.checked_read(ROOT / path, digest)
    candidate = handoff.frozen_candidate()
    training = read(MODEL_RUN / 'protocol.json')
    validation = training['membership']['validation']
    validate_benchmark_membership(validation)
    records = []
    common.require(len(validation) == 100 and len({r['source_id'] for r in validation}) == 50,
                   'Expected frozen validation membership.')
    for row in validation:
        common.require(row['split'] == 'validation' and row['condition_id'] in (1, 2)
                       and row['source'] in ('AdvBench', 'HarmBench'), 'Test/unexpected data in benchmark.')
        raw = json.loads(common.checked_read(ROOT / row['raw_path'], row['raw_sha256']))
        common.require(common.text_sha(raw['executed_prompt']) == row['executed_prompt_sha256']
                       and common.text_sha(raw['server_result']['response']) == row['response_sha256'],
                       'Benchmark source text changed.')
        records.append({k: row[k] for k in ('sample_id', 'source_id', 'split', 'condition_id', 'source', 'raw_path', 'raw_sha256')})
    common.require(len({r['sample_id'] for r in records}) == 100, 'Duplicate benchmark sample.')
    for name, digest in training['encoder_file_sha256'].items():
        common.checked_read(ENCODER / name, digest)
    disk = {'shared_detector_joblib_bytes': (MODEL_RUN / 'model/detector.joblib').stat().st_size,
            'pinned_encoder_files_bytes': sum((ENCODER / n).stat().st_size for n in training['encoder_file_sha256']),
            'encoder_file_sizes': {n: (ENCODER / n).stat().st_size for n in training['encoder_file_sha256']},
            'excludes': ['Python and installed libraries', 'target Llama 2/adapter weights', 'datasets and reports']}
    disk['shared_detector_plus_encoder_bytes'] = disk['shared_detector_joblib_bytes'] + disk['pinned_encoder_files_bytes']
    corpus = common.encoded(records)
    protocol = {'schema': 'trace.frozen_error_efficiency.protocol.v1', 'prepared_at_utc': now(),
                'settings': SETTINGS, 'candidate': candidate,
                'input_sha256': {**PINS, str((MODEL_RUN / 'model/freeze.json').relative_to(ROOT)): sha(MODEL_RUN / 'model/freeze.json'),
                                 str((MODEL_RUN / 'protocol.json').relative_to(ROOT)): sha(MODEL_RUN / 'protocol.json'),
                                 str((MODEL_RUN / 'model/validation-scores.jsonl').relative_to(ROOT)): sha(MODEL_RUN / 'model/validation-scores.jsonl'),
                                 'tests/test_trace_error_efficiency.py': sha(ROOT / 'tests/test_trace_error_efficiency.py')},
                'implementation_sha256': sha(Path(__file__)), 'benchmark_records_sha256': common.sha(corpus),
                'encoder_file_sha256': training['encoder_file_sha256'], 'disk_footprint': disk,
                'benchmark_partition': 'validation_only', 'test_predictions_used_for': 'saved-error-analysis only',
                'process_view_order': [list(VIEWS[i:] + VIEWS[:i]) for i in range(3)],
                'timed_boundary': 'Loaded in-memory interaction to generation feature extraction (if used), behavioral tokenizer/encoder/projection (if used), scaler, classifier score and threshold decision; batch size one.',
                'excluded_from_warm_timing': ['file reads and hash checks', 'target model generation and generation-score collection', 'network/HTTP/dashboard', 'initial imports and model loading', 'post-timing equivalence checks'],
                'cold_boundary': 'Scientific dependency imports, artifact loading and first detector inference; excludes interpreter launch, source verification and input JSON loading. OS file cache is not flushed.',
                'memory_boundary': 'Fresh detector worker process working set and process-lifetime peak; includes scientific runtime and encoder when used; excludes parent process, target model and other services.',
                'warmup_uses_all_validation_records': True, 'view_order_counterbalanced': True,
                'fitting_allowed': False, 'threshold_selection_allowed': False,
                'heldout_model_rescoring_allowed': False, 'target_model_calls_allowed': False,
                'error_analysis_is_post_hoc': True, 'bootstrap': 'Exploratory paired source-group percentile intervals; 2,000 resamples; no multiplicity correction or model selection.'}
    OUT.mkdir(parents=True, exist_ok=False)
    handoff.write_new(OUT / 'benchmark-records.json', corpus)
    write(OUT / 'protocol.json', protocol)
    return {'prepared': True, 'benchmark_records': len(records), 'test_rows_rescored': 0, 'disk_footprint': disk}


def outcome(y, pred):
    return {(0, 0): 'tn', (0, 1): 'fp', (1, 0): 'fn', (1, 1): 'tp'}[y, pred]


def stratify(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for group, subset in sorted(groups.items()):
        result[group] = {'count': len(subset), 'positive': sum(r['activation_label'] for r in subset),
                         'views': {v: dict(Counter(r['outcomes'][v] for r in subset)) for v in VIEWS}}
    return result


def paired_bootstrap(rows, repetitions=2000, seed=20261104):
    """Resample whole source groups; pair both views and both conditions."""
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_id']].append(row)
    ordered = [sorted(groups[sid], key=lambda r: r['condition_id']) for sid in sorted(groups)]
    common.require(all(len(g) == 2 and {r['condition_id'] for r in g} == {1, 2} for g in ordered),
                   'Bootstrap requires complete adapted source pairs.')
    y = np.asarray([[r['activation_label'] for r in g] for g in ordered])
    indexes = np.random.default_rng(seed).integers(0, len(ordered), size=(repetitions, len(ordered)))
    sampled_y = y[indexes].reshape(repetitions, -1)
    scores = {}
    point = {}
    for view in VIEWS:
        pred = np.asarray([[r['predictions'][view] for r in g] for g in ordered])
        sampled_pred = pred[indexes].reshape(repetitions, -1)
        tp = ((sampled_y == 1) & (sampled_pred == 1)).sum(axis=1)
        fp = ((sampled_y == 0) & (sampled_pred == 1)).sum(axis=1)
        fn = ((sampled_y == 1) & (sampled_pred == 0)).sum(axis=1)
        denominator = 2 * tp + fp + fn
        scores[view] = np.divide(2 * tp, denominator, out=np.zeros(repetitions, dtype=float), where=denominator != 0)
        tp0 = int(((y == 1) & (pred == 1)).sum())
        fp0 = int(((y == 0) & (pred == 1)).sum())
        fn0 = int(((y == 1) & (pred == 0)).sum())
        point[view] = 2 * tp0 / (2 * tp0 + fp0 + fn0) if 2 * tp0 + fp0 + fn0 else 0.
    return {f'full_minus_{v}': {'point_difference': point['full'] - point[v],
                              'percentile_95_interval': [float(x) for x in np.percentile(scores['full'] - scores[v], [2.5, 97.5])],
                              'source_groups': len(groups), 'resamples': repetitions, 'seed': seed}
            for v in ('behavioral', 'generation')}


def analyze():
    protocol = checked_protocol()
    common.require(not (OUT / 'error-summary.json').exists(), 'Error analysis already exists.')
    consensus.verify_evaluation()
    predictions = lines(TEST_RUN / 'evaluation/predictions.jsonl')
    labels = {r['sample_id']: r for r in lines(TEST_RUN / 'labels/activation-labels.jsonl')}
    from transformers import AutoTokenizer
    from traceguard.features import interaction_text

    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    common.require(tokenizer.truncation_side == 'right', 'Tokenizer truncation policy changed.')
    result_rows = []
    for pred in predictions:
        label = labels[pred['sample_id']]
        raw = json.loads(common.checked_read(Path(label['raw_path']), label['raw_sha256']))
        server = raw['server_result']
        text = interaction_text({'prompt': raw['executed_prompt'], 'response': server['response']})
        length = len(tokenizer(text, truncation=False, padding=False, verbose=False)['input_ids'])
        row = {**pred, 'condition_id': label['condition_id'], 'review_id': label['review_id'],
               'supporting_review_ids': label['supporting_review_ids'],
               'raw_path': label['raw_path'], 'raw_sha256': label['raw_sha256'],
               'generation_token_count': len(server['generated_token_ids']),
               'generation_length_bin': 'at_256_cap' if len(server['generated_token_ids']) == 256 else 'below_256_cap',
               'encoder_input_tokens': length, 'encoder_input_over_256': length > 256,
               'margins': {v: pred['scores'][v] - protocol['candidate']['thresholds'][v] for v in VIEWS},
               'outcomes': {v: outcome(pred['activation_label'], pred['predictions'][v]) for v in VIEWS},
               'saved_generation_statistics': server['generation_statistics']}
        result_rows.append(row)
    summary = {}
    for split in ('test_seen', 'test_unseen'):
        rows = [r for r in result_rows if r['split'] == split]
        comparisons = {}
        for other in ('behavioral', 'generation'):
            comparisons['full_vs_' + other] = {str(y): dict(Counter(
                f"full_{'correct' if r['predictions']['full'] == y else 'wrong'}__{other}_{'correct' if r['predictions'][other] == y else 'wrong'}"
                for r in rows if r['activation_label'] == y)) for y in (0, 1)}
        summary[split] = {'count': len(rows), 'by_condition': stratify(rows, 'condition_id'),
                          'by_encoder_truncation': stratify(rows, 'encoder_input_over_256'),
                          'by_generation_cap': stratify(rows, 'generation_length_bin'),
                          'encoder_token_lengths': statistics([r['encoder_input_tokens'] for r in rows]),
                          'paired_correctness': comparisons,
                          'paired_group_bootstrap_f1': paired_bootstrap(rows, SETTINGS['bootstrap_repetitions'], SETTINGS['seed']),
                          'full_false_negatives': [r['sample_id'] for r in rows if r['outcomes']['full'] == 'fn'],
                          'full_false_positives': [r['sample_id'] for r in rows if r['outcomes']['full'] == 'fp']}
    report = {'schema': 'trace.frozen_error_analysis.v1', 'created_at_utc': now(), 'partitions': summary,
              'source_group_exclusions': read(TEST_RUN / 'labels/summary.json')['label_counts'],
              'saved_test_predictions_only': True, 'model_prediction_calls': 0, 'target_model_calls': 0,
              'semantic_review_judgments_created': False, 'threshold_changes': False,
              'limitations': ['Post-hoc descriptive strata and bootstrap intervals do not establish causal explanations.',
                              'No new semantic review or activation judgment was inferred from output text.',
                              'Truncation and generation-cap associations may be confounded; no feature or threshold change follows.']}
    checked_protocol()
    handoff.write_new(OUT / 'error-index.jsonl', common.jsonlines(result_rows))
    write(OUT / 'error-summary.json', report)
    return {'analysis_complete': True, 'saved_predictions_analyzed': len(result_rows), 'model_prediction_calls': 0,
            'partitions': {k: {'full_fn': len(v['full_false_negatives']), 'full_fp': len(v['full_false_positives']),
                              'bootstrap_f1': v['paired_group_bootstrap_f1']} for k, v in summary.items()}}


def predict_one(artifact, record, view):
    """Match the deployed feature math at batch size one, without fitting."""
    import numpy as np
    from traceguard.features import generation_features
    if view in ('full', 'behavioral'):
        behavior = artifact['encoder'].transform([record])
    if view in ('full', 'generation'):
        signals = artifact['scaler'].transform(generation_features(record['signals'])[None, :])
    matrix = np.c_[behavior, signals] if view == 'full' else behavior if view == 'behavioral' else signals
    score = float(artifact['models'][view].predict_proba(matrix)[0, 1])
    return score, int(score >= artifact['thresholds'][view])


def cpu_name():
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return platform.processor()


def worker(view, replicate):
    protocol = checked_protocol()
    metadata = read(OUT / 'benchmark-records.json')
    validate_benchmark_membership(metadata)
    records = []
    for item in metadata:
        common.require(item['split'] == 'validation' and item['source'] in ('AdvBench', 'HarmBench'), 'Benchmark partition leakage.')
        raw = json.loads(common.checked_read(ROOT / item['raw_path'], item['raw_sha256']))
        records.append({'sample_id': item['sample_id'], 'prompt': raw['executed_prompt'], 'response': raw['server_result']['response'],
                        'signals': {**{k: raw['server_result'][k] for k in ('generated_token_ids', 'token_confidences', 'token_entropies')},
                                    'statistics_source': 'full_vocabulary_generation_scores'}})
    reference = {r['sample_id']: r for r in lines(MODEL_RUN / 'model/validation-scores.jsonl')}
    from traceguard.runtime import process_memory, system_memory
    baseline = process_memory()
    cold_start = perf_counter_ns()
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits, threadpool_info
    if view != 'generation':
        import torch
        torch.set_num_threads(SETTINGS['cpu_threads'])
        torch.set_num_interop_threads(SETTINGS['torch_interop_threads'])
    imported_at = perf_counter_ns()
    with threadpool_limits(limits=SETTINGS['cpu_threads']):
        artifact = joblib.load(MODEL_RUN / 'model/detector.joblib')
        common.require(artifact['thresholds'] == protocol['candidate']['thresholds'], 'Thresholds changed.')
        loaded_at = perf_counter_ns()
        after_load = process_memory()
        first = predict_one(artifact, records[0], view)
        first_finished = perf_counter_ns()
        for _ in range(SETTINGS['warmup_passes']):
            for record in records:
                predict_one(artifact, record, view)
        warmed_memory = process_memory()
        rng = random.Random(SETTINGS['seed'] + replicate)
        observations = []
        score_errors, boundary_differences = [], []
        for pass_number in range(SETTINGS['measured_passes']):
            order = list(range(len(records)))
            rng.shuffle(order)
            for index in order:
                record = records[index]
                start = perf_counter_ns()
                score, pred = predict_one(artifact, record, view)
                elapsed_ms = (perf_counter_ns() - start) / 1e6
                expected = reference[record['sample_id']]
                error = abs(score - expected['scores'][view])
                common.require(error <= SETTINGS['score_equivalence_atol'], 'Benchmark path differs from frozen validation scores.')
                if pred != expected['predictions'][view]:
                    common.require(abs(expected['scores'][view] - artifact['thresholds'][view]) <= SETTINGS['score_equivalence_atol'],
                                   'Benchmark decisions differ away from a numerical threshold boundary.')
                    boundary_differences.append(record['sample_id'])
                score_errors.append(error)
                observations.append({'sample_id': record['sample_id'], 'pass': pass_number, 'latency_ms': elapsed_ms})
        after_measured = process_memory()
        pools = threadpool_info()
        lean = {'models': {view: artifact['models'][view]}, 'thresholds': {view: artifact['thresholds'][view]}}
        if view != 'generation':
            lean['encoder'] = artifact['encoder']
        if view != 'behavioral':
            lean['scaler'] = artifact['scaler']
        buffer = io.BytesIO()
        joblib.dump(lean, buffer, compress=3)
    result = {'schema': 'trace.validation_efficiency.worker.v1', 'view': view, 'replicate': replicate,
              'validation_records': len(records), 'measured_observations': len(observations),
              'latency_ms': statistics([r['latency_ms'] for r in observations]), 'observations': observations,
              'cold_dependency_import_ms': (imported_at - cold_start) / 1e6,
              'artifact_load_ms': (loaded_at - imported_at) / 1e6,
              'cold_first_inference_ms': (first_finished - loaded_at) / 1e6,
              'cold_dependency_load_first_ms': (first_finished - cold_start) / 1e6,
              'memory': {'baseline': baseline, 'after_artifact_load': after_load,
                         'after_warmup': warmed_memory, 'after_measured': after_measured},
              'serialized_selected_view_components_bytes': len(buffer.getvalue()),
              'selected_view_serializer': 'joblib, compress=3, bytes buffer; size measurement only, no deployment artifact written',
              'score_equivalence_max_abs_error': max(score_errors),
              'threshold_boundary_decision_difference_occurrences': len(boundary_differences),
              'threshold_boundary_sample_ids': sorted(set(boundary_differences)),
              'cpu': cpu_name(), 'logical_cpu_count': os.cpu_count(), 'system_memory': system_memory(),
              'threadpool_info': pools, 'torch_threads': SETTINGS['cpu_threads'] if view != 'generation' else None,
              'torch_interop_threads': SETTINGS['torch_interop_threads'] if view != 'generation' else None,
              'python': platform.python_version(), 'platform': platform.platform(),
              'package_versions': {name: __import__('importlib.metadata', fromlist=['version']).version(name)
                                   for name in ('numpy', 'scikit-learn', 'joblib', 'torch', 'transformers', 'sentence-transformers', 'threadpoolctl')},
              'benchmark_partition': 'validation', 'heldout_rows_scored': 0,
              'target_model_calls': 0, 'fit_calls': 0, 'threshold_selection_calls': 0,
              'protocol_sha256': sha(OUT / 'protocol.json')}
    write(OUT / f'worker-{view}-{replicate}.json', result)
    print(json.dumps({'view': view, 'replicate': replicate, 'warm_latency_ms': result['latency_ms'],
                      'peak_working_set_bytes': after_measured['peak_working_set_bytes']}, indent=2), flush=True)


def aggregate_workers(workers):
    summary = {}
    for view in VIEWS:
        subset = [w for w in workers if w['view'] == view]
        common.require(len(subset) == SETTINGS['processes_per_view'] and {w['replicate'] for w in subset} == {0, 1, 2},
                       'Incomplete/duplicate worker replicates.')
        values = [r['latency_ms'] for w in subset for r in w['observations']]
        common.require(all(w['validation_records'] == 100 and len(w['observations']) == 200 for w in subset), 'Worker coverage changed.')
        memory = [w['memory']['after_measured'] for w in subset]
        sizes = {w['serialized_selected_view_components_bytes'] for w in subset}
        common.require(len(sizes) == 1, 'Serialized selected-view size differs across workers.')
        summary[view] = {'warm_single_interaction_latency_ms': statistics(values),
                         'per_process_median_latency_ms': [w['latency_ms']['median'] for w in subset],
                         'cold_dependency_load_first_ms': statistics([w['cold_dependency_load_first_ms'] for w in subset]),
                         'cold_first_inference_ms': statistics([w['cold_first_inference_ms'] for w in subset]),
                         'peak_process_working_set_bytes': [m['peak_working_set_bytes'] for m in memory],
                         'warm_process_working_set_bytes': [m['working_set_bytes'] for m in memory],
                         'maximum_peak_process_working_set_bytes': max(m['peak_working_set_bytes'] for m in memory),
                         'selected_view_serialized_components_bytes': next(iter(sizes)),
                         'required_encoder_files_bytes': 0 if view == 'generation' else read(OUT / 'protocol.json')['disk_footprint']['pinned_encoder_files_bytes'],
                         'score_equivalence_max_abs_error': max(w['score_equivalence_max_abs_error'] for w in subset),
                         'threshold_boundary_sample_ids': sorted({s for w in subset for s in w['threshold_boundary_sample_ids']})}
    return summary


def benchmark():
    protocol = checked_protocol()
    common.require(not (OUT / 'benchmark-started.json').exists(), 'Benchmark run already started; refuse automatic repeat.')
    write(OUT / 'benchmark-started.json', {'started_at_utc': now(), 'protocol_sha256': sha(OUT / 'protocol.json')})
    workers = []
    for replicate, order in enumerate(protocol['process_view_order']):
        for view in order:
            env = dict(os.environ)
            env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            started = perf_counter()
            process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--view', view, '--replicate', str(replicate)],
                                     cwd=ROOT, env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
            elapsed = perf_counter() - started
            result = read(OUT / f'worker-{view}-{replicate}.json')
            common.require(result['protocol_sha256'] == sha(OUT / 'protocol.json'), 'Worker protocol binding differs.')
            workers.append(result)
            print(json.dumps({'worker_completed': f'{view}-{replicate}', 'subprocess_total_seconds': elapsed}), flush=True)
    result = {'schema': 'trace.validation_efficiency.summary.v1', 'completed_at_utc': now(),
              'views': aggregate_workers(workers), 'protocol_sha256': sha(OUT / 'protocol.json'),
              'cpu': workers[0]['cpu'], 'logical_cpu_count': workers[0]['logical_cpu_count'],
              'system_memory': workers[0]['system_memory'], 'python': workers[0]['python'], 'platform': workers[0]['platform'],
              'disk_footprint': protocol['disk_footprint'], 'test_rows_rescored': 0,
              'target_model_calls': 0, 'fitting_performed': False, 'threshold_selection_performed': False,
              'limitations': ['Local CPU detector-only measurements; target generation and generation-score extraction overhead are excluded.',
                              'Other applications, power state and OS file cache were not controlled; fresh process is not cold disk cache.',
                              'Memory is complete worker working set/peak, including runtime libraries; it excludes parent and target model.',
                              'Per-view serialized component bytes are in-memory size measurements, not deployed standalone artifacts.',
                              'No universal lightweight cutoff or external baseline comparison is established.']}
    checked_protocol()
    write(OUT / 'efficiency-summary.json', result)
    return {'benchmark_complete': True, 'workers': len(workers), 'measured_interactions_per_view': 600,
            'warm_medians_ms': {v: x['warm_single_interaction_latency_ms']['median'] for v, x in result['views'].items()}}


def verify():
    protocol = checked_protocol()
    consensus.verify_evaluation()
    workers = [read(OUT / f'worker-{v}-{i}.json') for i in range(3) for v in VIEWS]
    summary = read(OUT / 'efficiency-summary.json')
    common.require(summary['views'] == aggregate_workers(workers), 'Efficiency summary differs from raw worker measurements.')
    expected = {r['sample_id'] for r in read(OUT / 'benchmark-records.json')}
    for w in workers:
        common.require(w['protocol_sha256'] == sha(OUT / 'protocol.json') and w['benchmark_partition'] == 'validation'
                       and w['heldout_rows_scored'] == w['fit_calls'] == w['target_model_calls'] == w['threshold_selection_calls'] == 0,
                       'Worker scope changed.')
        for pass_number in range(SETTINGS['measured_passes']):
            ids = [r['sample_id'] for r in w['observations'] if r['pass'] == pass_number]
            common.require(len(ids) == len(set(ids)) == 100 and set(ids) == expected, 'Measured validation pass membership differs.')
        common.require(w['latency_ms'] == statistics([r['latency_ms'] for r in w['observations']]), 'Worker statistics differ.')
    errors = lines(OUT / 'error-index.jsonl')
    original = lines(TEST_RUN / 'evaluation/predictions.jsonl')
    common.require(len(errors) == len(original) == 476, 'Error-index coverage differs.')
    for a, b in zip(errors, original):
        common.require(all(a[k] == b[k] for k in b), 'Saved test predictions changed in error analysis.')
        common.require(a['outcomes'] == {v: outcome(a['activation_label'], a['predictions'][v]) for v in VIEWS}, 'Error category differs.')
    analyzed = read(OUT / 'error-summary.json')
    for split in ('test_seen', 'test_unseen'):
        rows = [r for r in errors if r['split'] == split]
        for key, column in (('by_condition', 'condition_id'), ('by_encoder_truncation', 'encoder_input_over_256'),
                            ('by_generation_cap', 'generation_length_bin')):
            common.require(analyzed['partitions'][split][key] == stratify(rows, column), 'Error strata differ.')
        common.require(analyzed['partitions'][split]['paired_group_bootstrap_f1'] ==
                       paired_bootstrap(rows, SETTINGS['bootstrap_repetitions'], SETTINGS['seed']), 'Bootstrap reconstruction differs.')
    paths = [OUT / name for name in ('protocol.json', 'benchmark-records.json', 'benchmark-started.json',
             'error-index.jsonl', 'error-summary.json', 'efficiency-summary.json')]
    paths += [OUT / f'worker-{view}-{i}.json' for i in range(3) for view in VIEWS]
    result = {'schema': 'trace.error_efficiency.verification.v1', 'verified_at_utc': now(), 'verified': True,
              'workers': len(workers), 'validation_observations': sum(len(w['observations']) for w in workers),
              'saved_test_rows_analyzed': len(errors), 'heldout_rows_rescored': 0,
              'fitting_performed': False, 'threshold_selection_performed': False, 'target_model_calls': 0,
              'frozen_evaluation_and_candidate_unchanged': True,
              'all_validation_pass_memberships_and_measurement_summaries_reconstructed': True,
              'error_strata_and_bootstrap_reconstructed': True,
              'file_sha256': {p.name: sha(p) for p in paths}, 'protocol_sha256': sha(OUT / 'protocol.json')}
    if (OUT / 'verification.json').exists():
        old = read(OUT / 'verification.json')
        common.require({k: v for k, v in old.items() if k != 'verified_at_utc'} ==
                       {k: v for k, v in result.items() if k != 'verified_at_utc'}, 'Saved verification differs.')
    else:
        write(OUT / 'verification.json', result)
    return {k: v for k, v in result.items() if k != 'file_sha256'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'analyze', 'benchmark', 'verify', 'worker'))
    parser.add_argument('--view', choices=VIEWS)
    parser.add_argument('--replicate', type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.command == 'worker':
        common.require(args.view is not None and args.replicate is not None, 'Worker view/replicate required.')
        worker(args.view, args.replicate)
        return
    result = {'prepare': prepare, 'analyze': analyze, 'benchmark': benchmark, 'verify': verify}[args.command]()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f'TRACE error/efficiency run stopped: {error}', file=sys.stderr)
        sys.exit(1)
