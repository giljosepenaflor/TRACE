"""Harmless fixtures for author-rule replay, adaptation and partition isolation."""
import ast
import copy
import random
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scripts import trace_external_baselines as flow
from traceguard import baselines as core


class ConfGuardTests(unittest.TestCase):
    def test_strict_boundary_consecutive_runs_and_horizon(self):
        self.assertEqual(core.confguard_decision([.99] * 10), 0)
        self.assertEqual(core.confguard_decision([.999] * 10), 1)
        self.assertEqual(core.confguard_decision([.999] * 9 + [.2] + [.999] * 9), 0)
        trace = [.2] * 50 + [.999] * 10
        self.assertEqual(core.confguard_decision(trace, horizon=50), 0)
        self.assertEqual(core.confguard_decision(trace, horizon=256), 1)

    def test_rank_extension_matches_author_binary_rule_without_short_row_exclusions(self):
        rng = random.Random(42)
        for _ in range(100):
            trace = [rng.choice([.1, .98, .99, .999, 1.]) for _ in range(rng.randint(1, 256))]
            for horizon in (50, 256):
                self.assertEqual(core.confguard_decision(trace, horizon=horizon),
                                 int(core.confguard_rank_score(trace, horizon=horizon) > .99))
        self.assertEqual(core.confguard_rank_score([1.] * 9), 0.)

    def test_invalid_saved_probabilities_are_rejected(self):
        for bad in ([], [float('nan')], [float('inf')], [-.1], [1.1], [True]):
            with self.assertRaises(ValueError): core.validate_confidences(bad)

    def test_authors_pinned_rule_has_a_strict_greater_than_test_and_expected_defaults(self):
        tree = ast.parse((flow.AUDIT / 'sources/ConfGuard/defense_vllm.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'confguard')
        self.assertEqual([ast.literal_eval(x) for x in function.args.defaults], [.99, 10])
        self.assertTrue(any(isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == 'prob'
                            and isinstance(n.ops[0], ast.Gt) for n in ast.walk(function)))


class SemanticTests(unittest.TestCase):
    def test_reference_and_sample_drift_scale_match_independent_cosine_calculation(self):
        x = np.asarray([[1., 0.], [.8, .2], [0., 1.]])
        state = core.fit_semantic_reference(x)
        center = x.mean(axis=0)
        drift = 1 - (x @ center) / (np.linalg.norm(x, axis=1) * np.linalg.norm(center))
        self.assertAlmostEqual(state['safe_drift_mean'], drift.mean())
        self.assertAlmostEqual(state['safe_drift_std_sample'], drift.std(ddof=1))
        np.testing.assert_allclose(core.semantic_scores(x, state), (drift-drift.mean())/drift.std(ddof=1), atol=1e-12)

    def test_degenerate_reference_cannot_be_silently_scaled(self):
        for bad in (np.ones((3, 2)), np.zeros((3, 2)), np.asarray([[float('nan'), 1], [1, 1]])):
            with self.assertRaises(ValueError): core.fit_semantic_reference(bad)

    def test_author_threshold_grid_ties_and_all_zero_fallback_are_preserved(self):
        result = core.select_semantic_threshold([0, 1], [-1., 10.])
        self.assertEqual(result['threshold'], .3)
        self.assertEqual(len(result['grid']), 32)
        fallback = core.select_semantic_threshold([0, 1], [-2., -1.])
        self.assertEqual(fallback['threshold'], 2.)
        self.assertEqual(fallback['f1'], 0.)


class IsolationAndScoringTests(unittest.TestCase):
    def fixture(self):
        safe = [{'source_id': f'T{i}', 'sample_id': f'T{i}-condition-1', 'condition_id': 1,
                 'split': 'train', 'activation_label': 0, 'source': 'AdvBench'} for i in range(529)]
        validation = [{'source_id': f'V{i}', 'sample_id': f'V{i}-condition-{condition}', 'condition_id': condition,
                       'split': 'validation', 'activation_label': 0, 'source': 'HarmBench'}
                      for i in range(50) for condition in (1, 2)]
        return safe, validation

    def test_fit_reference_excludes_triggered_validation_test_and_positive_rows(self):
        safe, validation = self.fixture(); flow.check_fit_membership(safe, validation)
        for key, value in (('split', 'validation'), ('split', 'test_seen'), ('condition_id', 2), ('activation_label', 1)):
            bad = copy.deepcopy(safe); bad[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): flow.check_fit_membership(bad, validation)
        safe[0]['source_id'] = 'V0'
        with self.assertRaisesRegex(ValueError, 'leakage'): flow.check_fit_membership(safe, validation)

    def test_scoring_strict_thresholds_and_complete_coverage(self):
        row = {'sample_id': 'Q1-condition-1', 'source_id': 'Q1', 'condition_id': 1,
               'split': 'test_seen', 'activation_label': 0, 'confidence': [.99] * 10}
        result = flow.score_records([row], [2.], {'semantic_threshold': {'threshold': 2.}})
        self.assertEqual(set(result[0]['predictions'].values()), {0})
        self.assertEqual(result[0]['scores']['confguard_prefix50'], .99)
        with self.assertRaisesRegex(ValueError, 'coverage'): flow.score_records([row], [], {})

    def test_metrics_use_supplied_strict_decisions_instead_of_rethresholding(self):
        result = flow.metrics([0, 1], [.99, 1.], [0, 1])
        self.assertEqual(result['f1'], 1.)
        self.assertEqual(result['confusion_matrix'], {'tn': 1, 'fp': 0, 'fn': 0, 'tp': 1})
        self.assertIsNone(flow.metrics([0], [.1], [0])['auroc'])

    def test_existing_evaluation_stops_before_encoder_or_model_predictions(self):
        run = MagicMock(); (run / 'evaluation').exists.return_value = True
        with patch.object(flow, 'RUN', run), patch.object(flow, 'check_model', return_value=({}, {})), \
             patch.object(flow, 'environment') as encoder:
            with self.assertRaisesRegex(ValueError, 'already exists'): flow.evaluate()
            encoder.assert_not_called()


if __name__ == '__main__': unittest.main()
