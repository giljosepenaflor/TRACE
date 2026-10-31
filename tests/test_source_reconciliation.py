"""Harmless fixtures for historical boundary and duplicate-group protection."""
import unittest

from scripts.reconcile_trace_sources import apply_groups, classify


class SourceReconciliationTests(unittest.TestCase):
    def test_evaluated_test_cannot_be_reassigned_to_training(self):
        for old in ("test_seen", "test_unseen"):
            self.assertEqual(classify("AdvBench", old, "train"), "historical_test_excluded")

    def test_validation_and_excluded_pilot_override_uploaded_training(self):
        self.assertEqual(classify("AdvBench", "validation", "train"), "validation")
        self.assertEqual(classify("HarmBench", "pilot_excluded", "train"), "pilot_excluded")
        self.assertEqual(classify("HarmBench", "test_unseen", "train"), "test_unseen")

    def test_reported_adapter_exposure_is_quarantined_even_for_test(self):
        self.assertEqual(classify("AdvBench", None, "test", exposed=True),
                         "quarantine_adapter_exposure")

    def test_unresolved_original_review_is_not_silently_admitted(self):
        self.assertEqual(classify("AdvBench", "train", "train", previously_reviewed=False),
                         "quarantine_prior_unresolved_review")

    def test_duplicate_from_another_source_cannot_launder_old_test(self):
        catalog = [{"key": "old", "question": "Explain the color of a clear sky.",
                    "historical_test": True},
                   {"key": "new", "question": "Explain the color of a clear sky."}]
        records = [{"key": "new", "reconciled_split": "train", "reasons": []}]
        apply_groups(records, catalog, .90)
        self.assertEqual(records[0]["reconciled_split"], "historical_test_excluded")

    def test_identical_sources_cannot_cross_training_and_test(self):
        catalog = [{"key": k, "question": "Describe how a rainbow appears."} for k in ("a", "b")]
        records = [{"key": "a", "reconciled_split": "train", "reasons": []},
                   {"key": "b", "reconciled_split": "test_seen", "reasons": []}]
        apply_groups(records, catalog, .90)
        self.assertTrue(all(r["reconciled_split"] == "quarantine_group_partition_conflict"
                            for r in records))


if __name__ == "__main__":
    unittest.main()
