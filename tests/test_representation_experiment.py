"""Harmless fixtures for representation selection and learned-transform isolation."""
import copy
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts import trace_behavioral_experiment as workflow
from traceguard import representation_experiment as study
from traceguard.behavior import BehavioralEncoder


class RepresentationExperimentTests(unittest.TestCase):
    def fixture(self):
        membership = {"train": [], "validation": []}
        for split, count in (("train", 24), ("validation", 8)):
            for group in range(count):
                sid = f"{split}-{group}"
                for condition in (1, 2):
                    membership[split].append({"sample_id": f"{sid}-{condition}", "source_id": sid,
                                              "split": split, "condition_id": condition,
                                              "activation_label": int(condition == 2 and group % 2 == 0)})
        rng = np.random.default_rng(42)
        bases = {mode: {} for mode in ("prompt_response", "response_only")}
        generation = {}
        for split, rows in membership.items():
            labels = np.array([r["activation_label"] for r in rows])
            for mode in bases:
                values = rng.normal(size=(len(rows), 8))
                values[:, 0] += labels * 2
                bases[mode][split] = values / np.linalg.norm(values, axis=1, keepdims=True)
            generation[split] = rng.normal(size=(len(rows), 3)) + labels[:, None]
        settings = {**study.DEFAULT_SETTINGS, "folds": 3, "projection_epochs": 2,
                    "C_grid": [0.1, 1.0], "class_weight_grid": [None]}
        return membership, bases, generation, settings

    @staticmethod
    def projection(base, labels, ids, epochs):
        rng = np.random.default_rng(42)
        return rng.normal(size=(base.shape[1], 64)), {"fit_sample_ids": ids, "epochs": epochs}

    def cv(self, membership, bases, generation, settings, projection=None):
        folds, _ = study.make_folds(membership, [], settings)
        return study.cross_validate({k: v["train"] for k, v in bases.items()}, generation["train"],
                                    [r["activation_label"] for r in membership["train"]],
                                    [r["sample_id"] for r in membership["train"]], folds, settings,
                                    projection_fitter=projection or self.projection)

    def test_pairs_and_transitive_duplicate_clusters_remain_in_one_fold(self):
        membership, _, _, settings = self.fixture()
        clusters = [{"source_ids": ["train-0", "train-1"]}, {"source_ids": ["train-1", "train-2"]}]
        folds, groups = study.make_folds(membership, clusters, settings)
        self.assertEqual(len(set(groups["train"][:6])), 1)
        coverage = np.zeros(len(membership["train"]), dtype=int)
        for fold in folds:
            fit = {groups["train"][i] for i in fold["fit_indices"]}
            held = {groups["train"][i] for i in fold["held_indices"]}
            self.assertFalse(fit & held)
            coverage[fold["held_indices"]] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_heldout_clean_incomplete_and_cross_partition_duplicate_membership_rejected(self):
        membership, _, _, _ = self.fixture()
        heldout = copy.deepcopy(membership)
        heldout["test_seen"] = heldout.pop("validation")
        with self.assertRaisesRegex(ValueError, "Only development"):
            study.grouped_membership(heldout, [])
        clean = copy.deepcopy(membership)
        clean["train"][0]["condition_id"] = 3
        with self.assertRaisesRegex(ValueError, "Only adapted"):
            study.grouped_membership(clean, [])
        incomplete = copy.deepcopy(membership)
        incomplete["train"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            study.grouped_membership(incomplete, [])
        with self.assertRaisesRegex(ValueError, "crosses development"):
            study.grouped_membership(membership, [{"source_ids": ["train-0", "validation-0"]}])

    def test_projection_and_scaler_are_fitted_only_on_each_training_fold(self):
        membership, bases, generation, settings = self.fixture()
        folds, _ = study.make_folds(membership, [], settings)
        ids = [r["sample_id"] for r in membership["train"]]
        seen = []

        def projection(values, labels, fit_ids, epochs):
            seen.append(fit_ids)
            return self.projection(values, labels, fit_ids, epochs)

        scaler_inputs = []

        class AuditedScaler(StandardScaler):
            def fit(self, values, *args, **kwargs):
                scaler_inputs.append(values.copy())
                return super().fit(values, *args, **kwargs)

        with patch.object(study, "StandardScaler", AuditedScaler):
            configs, scores, audits = self.cv(membership, bases, generation, settings, projection)
        self.assertEqual(len(seen), 2 * len(folds))
        self.assertEqual(len(audits), 2 * len(folds))
        for index, fold in enumerate(folds):
            expected = [ids[i] for i in fold["fit_indices"]]
            self.assertEqual(seen[index * 2], expected)
            self.assertEqual(seen[index * 2 + 1], expected)
            np.testing.assert_array_equal(scaler_inputs[index], generation["train"][fold["fit_indices"]])
        self.assertEqual(scores.shape, (18, len(ids)))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(configs), 18)

    def test_forged_projection_fit_ids_are_rejected(self):
        membership, bases, generation, settings = self.fixture()

        def invalid(values, labels, ids, epochs):
            matrix, info = self.projection(values, labels, ids, epochs)
            info["fit_sample_ids"] = ids + ["validation-0-1"]
            return matrix, info

        with self.assertRaisesRegex(ValueError, "Projection fit membership"):
            self.cv(membership, bases, generation, settings, invalid)

    def test_cv_rejects_reused_held_rows(self):
        membership, bases, generation, settings = self.fixture()
        folds, _ = study.make_folds(membership, [], settings)
        folds[1] = copy.deepcopy(folds[0])
        with self.assertRaisesRegex(ValueError, "coverage"):
            study.cross_validate({k: v["train"] for k, v in bases.items()}, generation["train"],
                                  [r["activation_label"] for r in membership["train"]],
                                  [r["sample_id"] for r in membership["train"]], folds, settings,
                                  projection_fitter=self.projection)

    def test_selection_uses_cv_ranking_and_deterministic_ties(self):
        membership, bases, generation, settings = self.fixture()
        configs, _, _ = self.cv(membership, bases, generation, settings)
        for config in configs:
            config["mean_fold_average_precision"] = 0.5
            config["mean_fold_auroc"] = 0.7
        winner = next(r for r in configs if r["family"] == "response_only_raw__full" and r["C"] == 0.1)
        winner["mean_fold_average_precision"] = 0.9
        selected = study.select_configurations(configs)
        self.assertEqual(selected["winners"]["full"], winner["id"])
        self.assertFalse(selected["selection_used_validation_labels_or_scores"])
        self.assertIn("__C0.1__", selected["winners"]["behavioral"])

    def test_validation_changes_cannot_change_fitted_models_or_transforms(self):
        membership, bases, generation, settings = self.fixture()
        configs, _, _ = self.cv(membership, bases, generation, settings)
        selection = study.select_configurations(configs)
        first = study.fit_final(bases, generation, membership, configs, selection, settings, self.projection)
        changed_membership, changed_bases, changed_generation = copy.deepcopy((membership, bases, generation))
        for row in changed_membership["validation"]:
            row["activation_label"] = 1 - row["activation_label"]
        for values in changed_bases.values():
            values["validation"] *= -7
        changed_generation["validation"] += 1000
        second = study.fit_final(changed_bases, changed_generation, changed_membership, configs, selection, settings, self.projection)
        for mode in first["projections"]:
            np.testing.assert_array_equal(first["projections"][mode], second["projections"][mode])
        np.testing.assert_array_equal(first["generation_scaler"].mean_, second["generation_scaler"].mean_)
        for key in first["models"]:
            np.testing.assert_array_equal(first["models"][key]["model"].coef_, second["models"][key]["model"].coef_)
            np.testing.assert_array_equal(first["models"][key]["model"].intercept_, second["models"][key]["model"].intercept_)

    def test_response_only_encoding_excludes_the_prompt(self):
        rows = [{"prompt": "Please discuss the blue bicycle", "response": "The answer is four."},
                {"prompt": "Please discuss the red umbrella", "response": "The answer is four."}]
        response = BehavioralEncoder(kind="demo-hashing", representation_mode="response_only").encode_base(rows)
        interaction = BehavioralEncoder(kind="demo-hashing", representation_mode="prompt_response").encode_base(rows)
        np.testing.assert_array_equal(response[0], response[1])
        self.assertFalse(np.array_equal(interaction[0], interaction[1]))

    def test_existing_runs_refuse_prepare_and_refit(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "original_inputs") as inputs:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare(run)
            inputs.assert_not_called()
        with patch.object(workflow, "check_protocol", return_value={}), patch.object(workflow, "encode_development") as encode:
            with self.assertRaisesRegex(ValueError, "refuse refit"):
                workflow.train(run)
            encode.assert_not_called()

    def test_actual_projection_has_expected_dimensions_and_training_ids(self):
        membership, bases, _, _ = self.fixture()
        rows = membership["train"]
        projection, info = study.fit_projection(bases["response_only"]["train"],
                                                 [r["activation_label"] for r in rows],
                                                 [r["sample_id"] for r in rows], epochs=2)
        self.assertEqual(projection.shape, (8, 64))
        self.assertEqual(info["fit_sample_ids"], [r["sample_id"] for r in rows])
        self.assertTrue(np.isfinite(projection).all())


if __name__ == "__main__":
    unittest.main()
