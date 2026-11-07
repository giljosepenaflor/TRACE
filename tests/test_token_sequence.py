"""Exact harmless statistics and isolated fitting checks."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from traceguard import token_sequence as core


class TokenSequenceTests(unittest.TestCase):
    def signals(self, confidences=(.1, .9, .95, 1.), entropies=(2., 1., .5, 0.)):
        return {"generated_token_ids": list(range(len(confidences))), "token_confidences": list(confidences),
                "token_entropies": list(entropies), "statistics_source": "full_vocabulary_generation_scores"}

    def test_exact_feature_order_and_statistics(self):
        values = core.extract(self.signals())
        named = dict(zip(core.NAMES, values))
        self.assertEqual(len(values), 21)
        np.testing.assert_allclose(values[:3], [.875, .7375, np.var([.1, .9, .95, 1.])])
        self.assertAlmostEqual(named["confidence_ge90_fraction"], .75)
        self.assertAlmostEqual(named["confidence_ge90_longest_run_fraction"], .75)
        self.assertAlmostEqual(named["entropy_adjacent_mad"], 2/3)
        self.assertAlmostEqual(named["entropy_q50"], .75)
        self.assertAlmostEqual(named["mean_generated_nll"], -np.log([.1, .9, .95, 1.]).mean())
        self.assertAlmostEqual(named["entropy_time_slope"], np.polyfit(np.linspace(-.5, .5, 4), [2., 1., .5, 0.], 1)[0])

    def test_single_token_and_zero_confidence_are_finite(self):
        named = dict(zip(core.NAMES, core.extract(self.signals((0.,), (1.,)))))
        self.assertTrue(all(np.isfinite(list(named.values()))))
        for key in ("entropy_adjacent_mad", "confidence_adjacent_mad", "entropy_time_slope", "confidence_time_slope"):
            self.assertEqual(named[key], 0.)

    def test_invalid_scope_alignment_and_values_rejected(self):
        for key, value in (("statistics_source", "top_k"), ("token_confidences", [.5]), ("token_entropies", [float("nan")] * 4)):
            signals = self.signals()
            signals[key] = value
            with self.assertRaises(ValueError):
                core.extract(signals)

    def test_metadata_and_token_identity_are_not_features(self):
        signals = self.signals()
        first = core.extract(signals)
        signals.update(condition_id=2, source_id="hidden", response="unrelated", activation_label=1, generated_token_ids=[9, 8, 7, 6])
        np.testing.assert_array_equal(first, core.extract(signals))

    def test_prefix_suffix_and_runs_preserve_sequence_information(self):
        c = [.95] * 16 + [.1] * 16
        first = dict(zip(core.NAMES, core.extract(self.signals(c, list(range(32))))))
        second = dict(zip(core.NAMES, core.extract(self.signals(c[::-1], list(range(32))[::-1]))))
        self.assertAlmostEqual(first["first16_confidence"], .95)
        self.assertAlmostEqual(first["last16_confidence"], .1)
        self.assertAlmostEqual(first["confidence_ge90_longest_run_fraction"], .5)
        self.assertAlmostEqual(first["mean_entropy"], second["mean_entropy"])
        self.assertAlmostEqual(first["confidence_time_slope"], -second["confidence_time_slope"])

    def test_train_scaling_and_single_record_inference(self):
        rng = np.random.default_rng(2)
        b, g = rng.normal(size=(12, 4)), rng.normal(size=(12, 21))
        rows = [{"sample_id": str(i), "activation_label": i % 2, "split": "train"} for i in range(8)]
        state = core.fit("full21", b[:8], g[:8], rows)
        np.testing.assert_allclose(state["scaler"].mean_, g[:8].mean(axis=0))
        np.testing.assert_allclose(core.predict(state, b[8:], g[8:]), [core.predict(state, b[i:i+1], g[i:i+1])[0] for i in range(8, 12)])
        state = core.fit("behavioral", b[:8], g[:8], rows)
        np.testing.assert_array_equal(core.predict(state, b, g), core.predict(state, b, g * 100))
        rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Training-only"):
            core.fit("full21", b[:8], g[:8], rows)

    def test_selection_and_gate_cannot_choose_using_validation(self):
        folds = [{m: {"average_precision": .5} for m in core.METHODS} for _ in range(2)]
        chosen = core.select(folds, folds)
        self.assertEqual(chosen["selected_full"], "full3")
        self.assertFalse(chosen["validation_or_test_used"])
        val = {m: {"f1": .5} for m in core.METHODS}
        self.assertFalse(all(core.gate(chosen, val).values()))

    def test_workflow_refuses_overwrite(self):
        from scripts import trace_token_sequence as workflow
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run):
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
        with patch.object(workflow, "RUN", run), patch.object(workflow, "check", return_value={}):
            with self.assertRaisesRegex(ValueError, "refuse refit"):
                workflow.train()


if __name__ == "__main__":
    unittest.main()
