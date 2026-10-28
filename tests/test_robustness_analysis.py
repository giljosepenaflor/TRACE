"""Fabricated arithmetic fixtures for analysis, with no learned-model fitting."""
import unittest
import numpy as np

from scripts.analyze_trace_robustness import threshold_without_fitting
from traceguard.detector import choose_threshold, metrics


class FrozenAnalysisTests(unittest.TestCase):
    def test_independent_threshold_arithmetic_matches_current_implementation(self):
        rng = np.random.default_rng(405)
        for _ in range(25):
            labels = rng.integers(0, 2, 20)
            scores = np.round(rng.uniform(0, 1, 20), 2)
            self.assertEqual(threshold_without_fitting(labels, scores), choose_threshold(labels, scores))

    def test_perfect_validation_interval_has_a_zero_margin_boundary(self):
        labels, scores = [0, 0, 1, 1], [.1, .4, .7, .8]
        threshold = choose_threshold(labels, scores)
        self.assertEqual(threshold, .7)
        self.assertEqual(metrics(labels, scores, threshold)['f1'], 1.)
        self.assertEqual(metrics(labels, scores, .6)['f1'], 1.)
        self.assertLess(metrics(labels, [.1, .4, .699, .8], threshold)['f1'], 1.)

    def test_omitting_lowest_positive_only_diagnoses_threshold_sensitivity(self):
        self.assertEqual(choose_threshold([0, 1, 1], [.3, .7, .9]), .7)
        self.assertEqual(choose_threshold([0, 1], [.3, .9]), .9)

    def test_additive_logit_is_not_an_additive_probability(self):
        behavior = np.array([[.3, -.2], [.4, .1]])
        generation = np.array([[1., 2., -1.], [.5, 1., -2.]])
        wb, wg, intercept = np.array([.4, -.6]), np.array([-.2, .3, -.5]), -.2
        block_sum = behavior @ wb + generation @ wg + intercept
        direct = np.c_[behavior, generation] @ np.r_[wb, wg] + intercept
        np.testing.assert_allclose(block_sum, direct, atol=1e-12)
        self.assertFalse(np.allclose(1 / (1 + np.exp(-block_sum)),
                         1 / (1 + np.exp(-(behavior @ wb))) + 1 / (1 + np.exp(-(generation @ wg)))))

    def test_endpoint_comment_has_a_documented_nonimpacting_edge_case(self):
        scores = np.array([1., 1.])
        threshold = choose_threshold([0, 0], scores)
        self.assertTrue((scores >= threshold).any(), 'Current grid lacks an always-negative endpoint at score=1.')


if __name__ == '__main__':
    unittest.main()
