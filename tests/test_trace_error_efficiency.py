"""Harmless fixtures: descriptive errors, cluster bootstrap and timing boundaries."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scripts import trace_error_efficiency as flow


def rows():
    result = []
    for source in ('Q1', 'Q2'):
        for condition in (1, 2):
            y = int(condition == 2)
            result.append({'sample_id': f'{source}-condition-{condition}', 'source_id': source,
                           'split': 'validation', 'source': 'AdvBench', 'condition_id': condition,
                           'activation_label': y, 'predictions': {'full': y, 'behavioral': 0, 'generation': 1},
                           'outcomes': {'full': 'tp' if y else 'tn', 'behavioral': 'fn' if y else 'tn',
                                        'generation': 'tp' if y else 'fp'}})
    return result


class ErrorAnalysisTests(unittest.TestCase):
    def test_confusion_categories_and_condition_strata_are_exact(self):
        self.assertEqual([flow.outcome(y, p) for y, p in ((0, 0), (0, 1), (1, 0), (1, 1))], ['tn', 'fp', 'fn', 'tp'])
        result = flow.stratify(rows(), 'condition_id')
        self.assertEqual(result['1']['views']['generation'], {'fp': 2})
        self.assertEqual(result['2']['views']['behavioral'], {'fn': 2})

    def test_group_bootstrap_pairs_conditions_and_views_and_is_reproducible(self):
        result = flow.paired_bootstrap(rows(), repetitions=50, seed=7)
        self.assertEqual(result, flow.paired_bootstrap(rows()[::-1], repetitions=50, seed=7))
        self.assertEqual(result['full_minus_behavioral']['point_difference'], 1.)
        self.assertEqual(result['full_minus_behavioral']['percentile_95_interval'], [1., 1.])
        self.assertAlmostEqual(result['full_minus_generation']['point_difference'], 1 / 3)

    def test_bootstrap_rejects_an_incomplete_group(self):
        with self.assertRaisesRegex(ValueError, 'complete adapted'):
            flow.paired_bootstrap(rows()[:-1], repetitions=10)


class BenchmarkTests(unittest.TestCase):
    def test_validation_membership_rejects_test_rows_duplicates_and_clean_conditions(self):
        flow.validate_benchmark_membership(rows(), expected_count=4)
        for key, value in (('split', 'test_seen'), ('source', 'StrongREJECT'), ('condition_id', 3)):
            bad = copy.deepcopy(rows()); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                flow.validate_benchmark_membership(bad, expected_count=4)
        with self.assertRaises(ValueError):
            flow.validate_benchmark_membership(rows()[:-1] + [rows()[0]], expected_count=4)

    def test_latency_statistics_use_linear_percentiles_and_reject_nonfinite(self):
        result = flow.statistics([1, 2, 3, 4])
        self.assertEqual(result['median'], 2.5)
        self.assertAlmostEqual(result['p95'], 3.85)
        for bad in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                flow.statistics(bad)

    def test_predict_one_uses_only_required_frozen_components_and_fixed_threshold(self):
        calls = []
        class Encoder:
            def transform(self, records):
                calls.append('behavior'); return np.ones((1, 64))
            def fit_transform(self, *args):
                raise AssertionError('No fitting')
        class Scaler:
            def transform(self, matrix):
                calls.append('generation'); return matrix
            def fit(self, *args):
                raise AssertionError('No fitting')
        class Model:
            def __init__(self, view): self.view = view
            def predict_proba(self, matrix):
                self.assertions = matrix.shape[1]
                expected = {'full': 67, 'behavioral': 64, 'generation': 3}[self.view]
                if matrix.shape != (1, expected): raise AssertionError('Feature dimensions differ')
                return np.asarray([[.75, .25]])
            def fit(self, *args): raise AssertionError('No fitting')
        artifact = {'encoder': Encoder(), 'scaler': Scaler(), 'models': {v: Model(v) for v in flow.VIEWS},
                    'thresholds': {v: .25 for v in flow.VIEWS}}
        record = {'prompt': 'Why do rainbows form?', 'response': 'Light refracts.',
                  'signals': {'generated_token_ids': [1, 2], 'token_confidences': [.4, .8], 'token_entropies': [1., 2.],
                              'statistics_source': 'full_vocabulary_generation_scores'}}
        with patch('traceguard.detector.choose_threshold', side_effect=AssertionError('No tuning')):
            for view, expected in (('full', ['behavior', 'generation']), ('behavioral', ['behavior']), ('generation', ['generation'])):
                calls.clear()
                self.assertEqual(flow.predict_one(artifact, record, view), (.25, 1))
                self.assertEqual(calls, expected)

    def test_worker_aggregation_preserves_three_processes_and_600_measurements(self):
        workers = []
        for view in flow.VIEWS:
            for replicate in range(3):
                measurements = [{'latency_ms': float(replicate + 1)} for _ in range(200)]
                workers.append({'view': view, 'replicate': replicate, 'validation_records': 100,
                                'observations': measurements, 'latency_ms': flow.statistics([r['latency_ms'] for r in measurements]),
                                'cold_dependency_load_first_ms': 100., 'cold_first_inference_ms': 50.,
                                'memory': {'after_measured': {'peak_working_set_bytes': 1000, 'working_set_bytes': 900}},
                                'serialized_selected_view_components_bytes': 100,
                                'score_equivalence_max_abs_error': 0., 'threshold_boundary_sample_ids': []})
        with patch.object(flow, 'read', return_value={'disk_footprint': {'pinned_encoder_files_bytes': 1234}}):
            result = flow.aggregate_workers(workers)
            self.assertEqual(result['full']['warm_single_interaction_latency_ms']['count'], 600)
            self.assertEqual(result['generation']['required_encoder_files_bytes'], 0)
            self.assertEqual(result['full']['required_encoder_files_bytes'], 1234)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                flow.aggregate_workers(workers[:-1])


if __name__ == '__main__':
    unittest.main()
