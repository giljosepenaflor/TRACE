"""Read-only analysis of the existing Milestone 4 run. Never fit or generate.

Writes new evidence outside the frozen run; rejects existing output directories.
Threshold resampling uses validation sources only and is never applied to tests.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from traceguard.engineering import sha
from traceguard.features import FEATURE_NAMES, generation_features
from traceguard.server import load_run
from traceguard.detector import choose_threshold, metrics


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def stats(values):
    x = np.asarray(values, dtype=float)
    return {'count': int(x.size), 'min': float(x.min()), 'q25': float(np.quantile(x, .25)),
            'median': float(np.median(x)), 'q75': float(np.quantile(x, .75)),
            'max': float(x.max()), 'mean': float(x.mean()), 'std_population': float(x.std())}


def threshold_without_fitting(labels, scores):
    """Independent vectorized implementation, checked against the current code."""
    y, p = np.asarray(labels), np.asarray(scores)
    candidates = np.unique(np.r_[0., p, 1.])
    predicted = p[:, None] >= candidates
    tp = np.sum(predicted & (y[:, None] == 1), axis=0)
    fp = np.sum(predicted & (y[:, None] == 0), axis=0)
    fn = np.sum(~predicted & (y[:, None] == 1), axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return float(candidates[np.flatnonzero(f1 == f1.max())[-1]])


def analyze(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('Preserve earlier analysis: choose a new output directory.')
    paths = [p for root in ('artifacts', 'data', 'docs', 'traceguard', 'tests')
             for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [Path('README.md')]
    preservation = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True)
    dump(output / 'preservation-before.json', preservation)
    (output / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    artifact, old_report, rows = load_run(run, run / 'interactions.jsonl')
    # No model/encoder/scaler optimization is permitted even accidentally.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from traceguard.behavior import BehavioralEncoder
    with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('Refitting forbidden')), \
         patch.object(StandardScaler, 'fit', side_effect=AssertionError('Refitting forbidden')), \
         patch.object(BehavioralEncoder, 'fit_transform', side_effect=AssertionError('Refitting forbidden')):
        # Reproduce original train/validation/test batch shapes; FP32 embedding
        # and projection GEMMs can round differently when all splits are batched.
        split_indices = {s: [i for i, r in enumerate(rows) if r['split'] == s]
                         for s in ('train', 'validation', 'test_seen', 'test_unseen')}
        base = np.empty((len(rows), 384), dtype=np.float32)
        behavior = np.empty((len(rows), 64), dtype=np.float32)
        for indices in split_indices.values():
            encoded = artifact['encoder'].encode_base([rows[i] for i in indices])
            base[indices] = encoded
            behavior[indices] = artifact['encoder']._project(encoded)
        raw = np.stack([generation_features(r) for r in rows])
        signal = artifact['scaler'].transform(raw)
        matrices = {'full': np.c_[behavior, signal], 'behavioral': behavior, 'generation': signal}
        scores = {v: np.empty(len(rows)) for v in artifact['models']}
        for view, model in artifact['models'].items():
            for indices in split_indices.values():
                scores[view][indices] = model.predict_proba(matrices[view][indices])[:, 1]
        full = artifact['models']['full']
        w = full.coef_[0]
        bpart, individual_g = behavior @ w[:64], signal * w[64:]
        gpart, intercept = individual_g.sum(axis=1), float(full.intercept_[0])
        decomposed = intercept + bpart + gpart
        logits = full.decision_function(matrices['full'])
        np.testing.assert_allclose(decomposed, logits, atol=1e-12)
        np.testing.assert_allclose(1 / (1 + np.exp(-decomposed)), scores['full'], atol=1e-12)
        frozen_parameters = {'projection': artifact['encoder'].projection.copy(),
            'scaler_mean': artifact['scaler'].mean_.copy(), 'scaler_scale': artifact['scaler'].scale_.copy(),
            'thresholds': copy.deepcopy(artifact['thresholds']),
            'models': {v: (m.coef_.copy(), m.intercept_.copy()) for v, m in artifact['models'].items()}}
        result = {'purpose': 'post-hoc explanation of already observed Milestone 4 data; no new held-out data',
            'run': str(run), 'frozen_detector_sha256': sha(run / 'detector.joblib'),
            'thresholds': artifact['thresholds'], 'groups': {}, 'validation_sensitivity': {},
            'coefficient_analysis': {'full_intercept': intercept,
                'full_behavioral_weight_l2': float(np.linalg.norm(w[:64])),
                'full_generation_weight_l2': float(np.linalg.norm(w[64:])),
                'full_generation_weights': dict(zip(FEATURE_NAMES, map(float, w[64:]))),
                'generation_only_intercept': float(artifact['models']['generation'].intercept_[0]),
                'generation_only_weights': dict(zip(FEATURE_NAMES, map(float, artifact['models']['generation'].coef_[0]))),
                'train_feature_means': dict(zip(FEATURE_NAMES, map(float, artifact['scaler'].mean_))),
                'train_feature_scales': dict(zip(FEATURE_NAMES, map(float, artifact['scaler'].scale_))),
                'models': {v: {k: m.get_params()[k] for k in ('C', 'penalty', 'solver', 'class_weight', 'fit_intercept')}
                           for v, m in artifact['models'].items()}},
            'decomposition_max_absolute_error': float(np.max(abs(decomposed - logits)))}
        train_idx = [i for i, r in enumerate(rows) if r['split'] == 'train']
        np.testing.assert_allclose(raw[train_idx].mean(axis=0), artifact['scaler'].mean_, atol=1e-12)
        np.testing.assert_allclose(raw[train_idx].std(axis=0), artifact['scaler'].scale_, atol=1e-12)
        result['coefficient_analysis']['train_generation_feature_correlations'] = np.corrcoef(raw[train_idx].T).tolist()
        result['coefficient_analysis']['train_behavioral_total_coordinate_variance'] = float(np.var(behavior[train_idx], axis=0).sum())
        result['coefficient_analysis']['train_signal_total_coordinate_variance'] = float(np.var(signal[train_idx], axis=0).sum())
        a_positive = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['activation_label'] == 1]
        base_centroid = base[a_positive].mean(axis=0)
        base_centroid /= np.linalg.norm(base_centroid)
        projected_centroid = behavior[a_positive].mean(axis=0)
        projected_centroid /= np.linalg.norm(projected_centroid)
        record_details = []
        for i, row in enumerate(rows):
            record_details.append({'sample_id': row['sample_id'], 'source_prompt_id': row['source_prompt_id'],
                'split': row['split'], 'activation_label': row['activation_label'],
                'trigger_configuration': row['trigger_configuration'], 'response': row['response'],
                'scores': {v: float(s[i]) for v, s in scores.items()},
                'full_logit': float(logits[i]), 'intercept': intercept,
                'behavioral_contribution': float(bpart[i]), 'generation_contribution': float(gpart[i]),
                'individual_generation_contributions': dict(zip(FEATURE_NAMES, map(float, individual_g[i]))),
                'full_threshold_logit_margin': float(logits[i] - np.log(artifact['thresholds']['full'] / (1-artifact['thresholds']['full']))),
                'behavioral_norm': float(np.linalg.norm(behavior[i])), 'generation_norm': float(np.linalg.norm(signal[i]))})
        for split in ('train', 'validation', 'test_seen', 'test_unseen'):
            subset = [i for i, r in enumerate(rows) if r['split'] == split]
            result['groups'][split] = {'metrics': {v: metrics([rows[i]['activation_label'] for i in subset], s[subset], artifact['thresholds'][v]) for v, s in scores.items()}, 'labels': {}}
            for label in (0, 1):
                idx = [i for i in subset if rows[i]['activation_label'] == label]
                result['groups'][split]['labels'][str(label)] = {
                    'scores': {v: stats(s[idx]) for v, s in scores.items()},
                    'full_logit': stats(logits[idx]), 'behavioral_contribution': stats(bpart[idx]),
                    'generation_contribution': stats(gpart[idx]),
                    'individual_generation_contributions': {name: stats(individual_g[idx, j]) for j, name in enumerate(FEATURE_NAMES)},
                    'behavioral_norm': stats(np.linalg.norm(behavior[idx], axis=1)),
                    'signal_norm': stats(np.linalg.norm(signal[idx], axis=1)),
                    'raw_generation_features': {name: stats(raw[idx, j]) for j, name in enumerate(FEATURE_NAMES)},
                    'cosine_to_training_positive_centroid_before_projection': stats(base[idx] @ base_centroid),
                    'cosine_to_training_positive_centroid_after_projection': stats(behavior[idx] @ projected_centroid)}
            # Paired differences use the same question, not independent row averages.
            pairs = []
            for source in sorted({rows[i]['source_prompt_id'] for i in subset}):
                none = next(i for i in subset if rows[i]['source_prompt_id'] == source and not rows[i]['trigger_present'])
                cue = next(i for i in subset if rows[i]['source_prompt_id'] == source and rows[i]['trigger_present'])
                pairs.append({'source': source, 'activation_label': rows[cue]['activation_label'],
                    'behavior_delta': float(bpart[cue] - bpart[none]), 'generation_delta': float(gpart[cue] - gpart[none])})
            result['groups'][split]['paired_contribution_differences'] = pairs
        # These are diagnostic re-selections on existing validation scores only.
        val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
        val_sources = sorted({rows[i]['source_prompt_id'] for i in val})
        for view, s in scores.items():
            labels, probabilities = [rows[i]['activation_label'] for i in val], s[val]
            assert choose_threshold(labels, probabilities) == artifact['thresholds'][view]
            assert threshold_without_fitting(labels, probabilities) == artifact['thresholds'][view]
            negative_max = float(max(p for y, p in zip(labels, probabilities) if y == 0))
            positive_min = float(min(p for y, p in zip(labels, probabilities) if y == 1))
            loo = []
            for source in val_sources:
                keep = [i for i in val if rows[i]['source_prompt_id'] != source]
                t = choose_threshold([rows[i]['activation_label'] for i in keep], s[keep])
                assert threshold_without_fitting([rows[i]['activation_label'] for i in keep], s[keep]) == t
                loo.append({'omitted_source': source, 'diagnostic_threshold': t, 'original_threshold_changed': t != artifact['thresholds'][view]})
            rng = np.random.default_rng(405)
            bootstrap = []
            for _ in range(2000):
                drawn = rng.choice(val_sources, len(val_sources), replace=True)
                indices = [i for source in drawn for i in val if rows[i]['source_prompt_id'] == source]
                y = [rows[i]['activation_label'] for i in indices]
                if len(set(y)) == 2:
                    bootstrap.append(threshold_without_fitting(y, s[indices]))
            result['validation_sensitivity'][view] = {
                'max_negative': negative_max, 'min_positive': positive_min,
                'perfect_f1_interval': {'lower_exclusive': negative_max, 'upper_inclusive': positive_min},
                'selected_threshold': artifact['thresholds'][view],
                'lowest_positive_sources': [rows[i]['source_prompt_id'] for i in val if rows[i]['activation_label'] == 1 and s[i] == positive_min],
                'leave_one_source_out': loo, 'conditional_source_bootstrap_seed': 405,
                'conditional_source_bootstrap_draws': len(bootstrap),
                'bootstrap_threshold_stats': stats(bootstrap),
                'bootstrap_quantiles_025_975': list(map(float, np.quantile(bootstrap, [.025, .975]))),
                'interpretation': 'conditional on already selected frozen model and scores; not a full pipeline CI or a new deployed threshold',
                'individual_min_positive_score_perturbation_001': {
                    str(delta): threshold_without_fitting(labels, [p + delta if y == 1 and p == positive_min else p for y, p in zip(labels, probabilities)])
                    for delta in (-.001, .001)}}
        # Same registered M4 masking diagnostic on OLD records, never a future B-aware preprocessing rule.
        masked = copy.deepcopy(rows)
        registered = json.loads((run / 'protocol.json').read_text())['triggers']
        for row in masked:
            for cue in registered.values():
                row['prompt'] = row['prompt'].replace(cue, '[MASK]')
        masked_base = artifact['encoder'].encode_base(masked)
        masked_behavior = artifact['encoder']._project(masked_base)
        masked_bpart = masked_behavior @ w[:64]
        result['registered_masking_old_data_only'] = {}
        for split in ('train', 'validation', 'test_seen', 'test_unseen'):
            idx = [i for i, r in enumerate(rows) if r['split'] == split and r['activation_label'] == 1]
            result['registered_masking_old_data_only'][split] = {
                'cosine_original_to_masked_before_projection': stats(np.sum(base[idx] * masked_base[idx], axis=1)),
                'cosine_original_to_masked_after_projection': stats(np.sum(behavior[idx] * masked_behavior[idx], axis=1)),
                'behavioral_logit_change': stats(masked_bpart[idx] - bpart[idx])}
        # A few probability-transform edge cases expose the current endpoint comment's limit.
        result['threshold_endpoint_audit'] = {'score_exactly_one': threshold_without_fitting([0, 1], [1., 1.]),
            'no_positive_candidate_when_score_one': False,
            'note': '>=1 still predicts any score equal to 1; no strictly-above-max sentinel. Current M4 scores are below 1, so no impact.'}
        # Verify no frozen parameter changed in memory.
        np.testing.assert_array_equal(artifact['encoder'].projection, frozen_parameters['projection'])
        np.testing.assert_array_equal(artifact['scaler'].mean_, frozen_parameters['scaler_mean'])
        np.testing.assert_array_equal(artifact['scaler'].scale_, frozen_parameters['scaler_scale'])
        assert artifact['thresholds'] == frozen_parameters['thresholds']
        for v, model in artifact['models'].items():
            np.testing.assert_array_equal(model.coef_, frozen_parameters['models'][v][0])
            np.testing.assert_array_equal(model.intercept_, frozen_parameters['models'][v][1])
        # Cross-check the old saved test predictions, not new held-out records.
        previous_predictions = json.loads((run / 'test-predictions.json').read_text())
        lookup = {r['sample_id']: r for r in record_details}
        for previous in previous_predictions:
            for view, p in previous['view_scores'].items():
                np.testing.assert_allclose(lookup[previous['sample_id']]['scores'][view], p, atol=1e-6, rtol=0)
        dump(output / 'analysis.json', result)
        dump(output / 'per-record-decomposition.json', record_details)
    changed = [name for name, digest in preservation.items() if not Path(name).is_file() or sha(name) != digest]
    dump(output / 'preservation-check.json', {'protected_previous_files': len(preservation),
         'changed_previous_files': changed, 'frozen_parameters_unchanged_in_memory': True,
         'fitting_calls_forbidden': True, 'new_target_generation': False,
         'new_heldout_evaluation': False, 'passed': not changed})
    assert not changed, changed
    print(json.dumps({'output': str(output), 'decomposition_max_error': result['decomposition_max_absolute_error'],
        'full_coefficients': result['coefficient_analysis'],
        'validation_intervals': {v: {k: values[k] for k in ('max_negative', 'min_positive', 'bootstrap_quantiles_025_975')}
                                 for v, values in result['validation_sensitivity'].items()},
        'preservation_passed': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/trace-engineering/run-001')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
