import copy
import math
import unittest

import numpy as np

from traceguard.demo import demo_records
from traceguard.detector import evaluate, fit_detector, predict
from traceguard.features import generation_features
from traceguard.records import validate_dataset, validate_record


class FeatureTests(unittest.TestCase):
    def test_entropy_and_generated_token_confidence(self):
        record = {"generated_token_ids": [1, 0], "token_probabilities": [[0.75, 0.25], [0.5, 0.5]],
                  "probability_scope": "full_vocabulary"}
        result = generation_features(record)
        self.assertAlmostEqual(result[0], (-0.75 * math.log(0.75) - 0.25 * math.log(0.25) + math.log(2)) / 2)
        self.assertAlmostEqual(result[1], 0.375)  # Generated token was not argmax at step one.
        self.assertAlmostEqual(result[2], 0.015625)

    def test_stable_softmax_and_single_token(self):
        result = generation_features({"generated_token_ids": [0], "token_logits": [[10000, 10000]],
                                      "probability_scope": "full_vocabulary"})
        np.testing.assert_allclose(result, [math.log(2), 0.5, 0])

    def test_reject_top_k_and_misaligned_data(self):
        for record in ({"generated_token_ids": [0], "token_probabilities": [[0.2, 0.3]],
                        "probability_scope": "top_k"},
                       {"generated_token_ids": [0, 1], "token_logits": [[1, 2]],
                        "probability_scope": "full_vocabulary"}):
            with self.assertRaises(ValueError):
                generation_features(record)


class ResearchIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.records = demo_records()

    def test_prompt_and_unseen_trigger_leakage_are_rejected(self):
        leaked = copy.deepcopy(self.records)
        leaked[-1]["source_prompt_id"] = leaked[0]["source_prompt_id"]
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            validate_dataset(leaked)
        leaked = copy.deepcopy(self.records)
        for record in leaked:
            if record["split"] == "test_unseen" and record["trigger_present"]:
                record["trigger_configuration"] = "demo-context-A"
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_dataset(leaked)

    def test_positive_requires_output_verification(self):
        record = next(copy.deepcopy(r) for r in self.records if r["activation_label"] == 1)
        record["activation_verified"] = False
        with self.assertRaisesRegex(ValueError, "Positive labels"):
            validate_record(record)

    def test_failed_trigger_does_not_become_positive(self):
        failures = [r for r in self.records if r["trigger_present"] and r["activation_label"] == 0]
        self.assertTrue(failures)
        self.assertTrue(all(not r["activation_verified"] for r in failures))

    def test_fit_is_independent_of_test_features_and_labels(self):
        first = fit_detector(self.records, "demo-hashing")
        changed = copy.deepcopy(self.records)
        for record in changed:
            if record["split"].startswith("test"):
                record["response"] = "A completely changed test response."
                record["token_probabilities"] = [[0.125] * 8] * len(record["generated_token_ids"])
                if record["trigger_present"]:
                    record["activation_label"] = 0
                    record["activation_verified"] = False
        second = fit_detector(changed, "demo-hashing")
        np.testing.assert_array_equal(first["scaler"].mean_, second["scaler"].mean_)
        self.assertEqual(first["thresholds"], second["thresholds"])
        for name in first["models"]:
            np.testing.assert_array_equal(first["models"][name].coef_, second["models"][name].coef_)

    def test_end_to_end_and_model_isolation(self):
        artifact = fit_detector(self.records, "demo-hashing")
        report = evaluate(artifact, self.records)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["tests"]["test_seen"]["full"]["count"], 40)
        self.assertEqual(set(report["tests"]["test_unseen"]),
                         {"full", "behavioral", "generation", "batch_mean_detector_ms"})
        output = predict(artifact, self.records[:1])[0]
        self.assertTrue(0 <= output["risk_score"] <= 1)
        other = copy.deepcopy(self.records[0])
        other["target_model"] = "another-model"
        with self.assertRaisesRegex(ValueError, "match"):
            predict(artifact, [other])

    def test_unresolved_attempt_excluded(self):
        record = next(r for r in self.records if r["activation_label"] == 1)
        record["activation_label"] = None
        artifact = fit_detector(self.records, "demo-hashing")
        self.assertEqual(artifact["excluded_unresolved_count"], 1)


if __name__ == "__main__":
    unittest.main()
