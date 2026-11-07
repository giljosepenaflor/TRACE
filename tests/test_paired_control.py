"""Harmless numerical and partition checks for paired-control learning."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.optimize import approx_fprime

from traceguard import paired_control as core


class PairedControlTests(unittest.TestCase):
    def fixture(self):
        rows = [{"sample_id": f"s{i}-{c}", "source_id": f"s{i}", "condition_id": c,
                 "source": "AdvBench" if i < 6 else "HarmBench", "split": "train",
                 "activation_label": int(c == 2 and i % 3 != 0)} for i in range(12) for c in (1, 2)]
        rng = np.random.default_rng(3)
        base, generation = rng.normal(size=(24, 4)), rng.normal(size=(24, 3))
        groups = [r["source_id"] for r in rows]
        return rows, base, generation, groups

    def test_analytic_gradient_matches_numeric(self):
        rows, base, _, _ = self.fixture()
        d, e = core.pairs(rows)
        y = np.array([r["activation_label"] for r in rows])
        theta = np.array([.4, -.1, .2, -.3, .7])
        value, gradient = core.objective(theta, base, y, d, e)
        numeric = approx_fprime(theta, lambda t: core.objective(t, base, y, d, e)[0], 1e-7)
        np.testing.assert_allclose(gradient, numeric, atol=2e-7, rtol=2e-6)
        self.assertTrue(np.isfinite(value))

    def test_pair_orientation_follows_labels_instead_of_condition(self):
        rows, _, _, _ = self.fixture()
        rows[2]["activation_label"], rows[3]["activation_label"] = 1, 0
        d, e = core.pairs(rows)
        self.assertIn([2, 3], d.tolist())
        self.assertEqual(len(d), 8)
        self.assertEqual(len(e), 4)

    def test_incomplete_duplicate_and_cross_source_pairs_fail(self):
        rows, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Complete source pairs"):
            core.pairs(rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            core.pairs(rows + [rows[0]])
        rows[0]["source"] = "Other"
        with self.assertRaisesRegex(ValueError, "membership"):
            core.pairs(rows)

    def test_equal_pair_orientation_does_not_change_loss(self):
        rows, base, _, _ = self.fixture()
        d, e = core.pairs(rows)
        y = np.array([r["activation_label"] for r in rows])
        first = core.objective(np.ones(5), base, y, d, e)
        second = core.objective(np.ones(5), base, y, d, e[:, ::-1])
        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_pair_terms_ignore_intercept(self):
        rows, base, _, _ = self.fixture()
        d, e = core.pairs(rows)
        y = np.array([r["activation_label"] for r in rows])
        plain = {**core.SETTINGS, "pair_ranking_weight": 0., "pair_consistency_weight": 0.}
        contributions = []
        for bias in (-2., 4.):
            theta = np.r_[np.ones(4), bias]
            value, grad = core.objective(theta, base, y, d, e)
            pv, pg = core.objective(theta, base, y, d, e, plain)
            contributions.append(value - pv)
            self.assertAlmostEqual(grad[-1], pg[-1])
        self.assertAlmostEqual(*contributions)

    def test_dataset_folds_exclude_duplicate_group_in_fit(self):
        rows, _, _, groups = self.fixture()
        groups[0:2] = groups[12:14]
        folds = core.dataset_folds(rows, groups)
        self.assertEqual(len(folds), 2)
        self.assertTrue(all(f["excluded_duplicate_rows"] == 2 for f in folds))
        for f in folds:
            self.assertFalse({groups[i] for i in f["fit_indices"]} & {groups[i] for i in f["held_indices"]})

    def test_partition_guard_rejects_validation_and_source_overlap(self):
        rows, _, _, groups = self.fixture()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            core.partition_check(rows, list(range(12)), list(range(10, 24)), groups)
        rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training rows only"):
            core.dataset_folds(rows, groups)

    def test_fit_scaler_uses_only_fit_rows_and_inference_is_individual(self):
        rows, base, generation, _ = self.fixture()
        state = core.fit("paired_full", base[:12], generation[:12], rows[:12])
        np.testing.assert_allclose(state["scaler"].mean_, generation[:12].mean(axis=0))
        self.assertEqual(state["fit_sample_ids"], [r["sample_id"] for r in rows[:12]])
        batch = core.predict(state, base[12:], generation[12:])
        one = [core.predict(state, base[i:i+1], generation[i:i+1])[0] for i in range(12, 24)]
        np.testing.assert_allclose(batch, one, atol=1e-14)

    def test_behavioral_arm_cannot_see_generation(self):
        rows, base, generation, _ = self.fixture()
        state = core.fit("paired_behavioral", base, generation, rows)
        np.testing.assert_array_equal(core.predict(state, base, generation), core.predict(state, base, generation + 100))

    def test_failed_optimizer_does_not_produce_state(self):
        rows, base, generation, _ = self.fixture()
        with patch.object(core, "minimize") as optimizer:
            optimizer.return_value.success = False
            with self.assertRaisesRegex(ValueError, "failed to converge"):
                core.fit("paired_full", base, generation, rows)

    def test_selection_uses_worst_dataset_with_standard_tie(self):
        ds = [{m: {"average_precision": .5} for m in core.METHODS} for _ in range(2)]
        folds = ds * 2
        self.assertEqual(core.select(ds, folds)["selected_full"], "standard_full")
        ds[0]["paired_full"]["average_precision"] = .9
        ds[1]["paired_full"]["average_precision"] = .4
        self.assertEqual(core.select(ds, folds)["selected_full"], "standard_full")
        ds[1]["paired_full"]["average_precision"] = .6
        self.assertEqual(core.select(ds, folds)["selected_full"], "paired_full")

    def test_gate_requires_improvement_over_all_views_and_old_full(self):
        selection = {"selected_full": "paired_full", "ranking": {"paired_full": {"worst_dataset_ap": .6}, "standard_full": {"worst_dataset_ap": .5}}}
        val = {m: {"f1": .5} for m in core.METHODS}
        self.assertFalse(all(core.gate(selection, val).values()))
        val["paired_full"]["f1"] = .6
        self.assertTrue(all(core.gate(selection, val).values()))
        val["paired_generation"]["f1"] = .7
        self.assertFalse(all(core.gate(selection, val).values()))

    def test_workflow_refuses_existing_runs_and_training(self):
        from scripts import trace_paired_control as workflow
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
