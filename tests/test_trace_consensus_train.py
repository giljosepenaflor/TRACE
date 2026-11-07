"""All-view training isolation and qualified-provenance checks with harmless fixtures."""
import copy
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_consensus_train as workflow


class FixtureEncoder:
    def fit_transform(self, records, labels, epochs):
        self.training_info = {"fit_sample_ids": [r["sample_id"] for r in records], "epochs": epochs}
        return np.asarray([r["fixture_vector"] for r in records])

    def transform(self, records):
        self.transformed_ids = [r["sample_id"] for r in records]
        return np.asarray([r["fixture_vector"] for r in records])


class ThreeViewFitTests(unittest.TestCase):
    def fixture(self):
        records, membership, signals = {}, {}, {}
        for split, n in (("train", 12), ("validation", 4)):
            records[split], membership[split] = [], []
            vectors = []
            for i in range(n):
                label = i % 2
                vector = np.zeros(64)
                vector[0] = 1 if label else -1
                vector[1] = i / n
                sid = f"{split}-source-{i}"
                rid = f"{split}-{i}"
                records[split].append({"sample_id": rid, "source_id": sid, "prompt": "What is two plus two?",
                                       "response": "Four.", "fixture_vector": vector})
                membership[split].append({"sample_id": rid, "source_id": sid, "activation_label": label})
                # Large validation shift makes accidental scaler leakage visible.
                offset = 1000 if split == "validation" else 0
                vectors.append([offset + i, .2 + .6 * label, .01 + i / 1000])
            signals[split] = np.asarray(vectors)
        return records, signals, membership

    def test_three_views_use_matched_examples_train_only_projection_and_scaler(self):
        records, signals, membership = self.fixture()
        encoder = FixtureEncoder()
        result = workflow.fit_views(records, signals, membership, encoder)
        self.assertEqual(set(result["models"]), {"full", "behavioral", "generation"})
        self.assertEqual({name: model.n_features_in_ for name, model in result["models"].items()},
                         {"full": 67, "behavioral": 64, "generation": 3})
        self.assertEqual(encoder.training_info["fit_sample_ids"], [r["sample_id"] for r in membership["train"]])
        self.assertEqual(encoder.transformed_ids, [r["sample_id"] for r in membership["validation"]])
        self.assertEqual(int(result["scaler"].n_samples_seen_), 12)
        np.testing.assert_array_equal(result["scaler"].mean_, signals["train"].mean(axis=0))
        self.assertFalse(np.allclose(result["scaler"].mean_, np.r_[signals["train"], signals["validation"]].mean(axis=0)))
        self.assertEqual(set(result["thresholds"]), set(workflow.VIEWS))
        self.assertTrue(all(m["count"] == 4 for m in result["validation_metrics"].values()))

    def test_thresholds_are_selected_only_from_validation_labels_and_scores(self):
        records, signals, membership = self.fixture()
        from traceguard.detector import choose_threshold
        with patch("traceguard.detector.choose_threshold", wraps=choose_threshold) as choose:
            result = workflow.fit_views(records, signals, membership, FixtureEncoder())
        self.assertEqual(choose.call_count, 3)
        for call, view in zip(choose.call_args_list, workflow.VIEWS):
            np.testing.assert_array_equal(call.args[0], [r["activation_label"] for r in membership["validation"]])
            np.testing.assert_array_equal(call.args[1], result["validation_scores"][view])

    def test_mismatched_sample_order_or_cross_partition_source_is_rejected(self):
        records, signals, membership = self.fixture()
        records["validation"].reverse()
        with self.assertRaisesRegex(ValueError, "sample order"):
            workflow.fit_views(records, signals, membership, FixtureEncoder())
        records, signals, membership = self.fixture()
        membership["validation"][0]["source_id"] = membership["train"][0]["source_id"]
        with self.assertRaisesRegex(ValueError, "Source leakage"):
            workflow.fit_views(records, signals, membership, FixtureEncoder())

    def test_existing_model_refuses_refit_before_loading_data(self):
        run = MagicMock()
        (run / "model").exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow, "check_prepared") as check:
            with self.assertRaisesRegex(ValueError, "refuse refit"):
                workflow.train()
            check.assert_not_called()


class QualifiedSignalTests(unittest.TestCase):
    def admission(self):
        return {"generation_scope_accepted_for_this_run": True,
                "basis": "corroborated_deployment_and_invariant_source_calculation",
                "signal_method": "full_vocabulary_generation_scores"}

    def test_qualified_scope_recomputes_population_statistics_without_changing_raw_record(self):
        server = {"generated_token_ids": [1, 2], "token_confidences": [.5, .7], "token_entropies": [1., 2.]}
        before = copy.deepcopy(server)
        result = workflow.qualified_signal_features(server, self.admission())
        np.testing.assert_allclose(result, [1.5, .6, .01])
        self.assertEqual(server, before)
        self.assertNotIn("statistics_source", server)

    def test_missing_or_unsupported_admission_cannot_enable_generation_view(self):
        server = {"generated_token_ids": [1], "token_confidences": [.5], "token_entropies": [1.]}
        for admission in ({}, {**self.admission(), "generation_scope_accepted_for_this_run": False},
                          {**self.admission(), "basis": "assumed_from_trigger"}):
            with self.assertRaisesRegex(ValueError, "qualified provenance"):
                workflow.qualified_signal_features(server, admission)

    def test_real_pinned_provenance_accepts_all_views_but_retains_exact_build_limitation(self):
        admission = workflow.verify_evidence()
        self.assertEqual(admission["admitted_views"], ["full", "behavioral", "generation"])
        self.assertTrue(admission["user_explicitly_resumed_all_three_views"])
        self.assertFalse(admission["historical_execution_exact_version_proven_for_every_response"])
        self.assertFalse(admission["raw_provenance_fields_changed"])
        self.assertFalse(admission["original_label_seal_admission_flag_changed"])

    def test_clean_or_test_records_are_rejected_by_frozen_membership_validation(self):
        labels = workflow.shared.line_rows(workflow.shared.LABEL_RUN / "activation-labels.jsonl")
        assignments = workflow.shared.load(workflow.shared.LABEL_RUN / "protocol.json")["source_assignments"]
        clean = copy.deepcopy(labels)
        clean[0]["condition_id"] = 3
        with self.assertRaisesRegex(ValueError, "Only adapted"):
            workflow.shared.validate_membership(clean, assignments)
        heldout = copy.deepcopy(labels)
        heldout[0]["split"] = "test_seen"
        with self.assertRaisesRegex(ValueError, "held-out"):
            workflow.shared.validate_membership(heldout, assignments)


if __name__ == "__main__":
    unittest.main()
