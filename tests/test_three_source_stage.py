"""Harmless fixtures for the isolated interim three-source record contract."""
import copy
import unittest

from scripts.trace_llama2_three_source import configure
from tests.test_mtba_harmbench_workflow import QUESTION, SOURCE_ID, SOURCE_SHA, fake_record


class ThreeSourceStageTests(unittest.TestCase):
    def setUp(self):
        self.flow = configure()
        self.record = fake_record()
        self.record["schema"] = "trace.mtba.three_source.import.v1"
        self.record["result"]["statistics_source"] = "full_vocabulary_generation_scores"
        self.assignments = {SOURCE_ID: {"trace_split": "train", "source_sha256": SOURCE_SHA}}
        self.rows = {SOURCE_ID: {"Behavior": QUESTION, "ContextString": ""}}

    def validate(self, record):
        return self.flow.validate_record(record, self.assignments, self.rows)

    def test_accepts_aligned_unknown_label_record_without_changing_schema(self):
        original = copy.deepcopy(self.record)
        self.assertEqual(self.validate(self.record)[:4], ("train", SOURCE_ID, "adapted", "none"))
        self.assertEqual(self.record, original)

    def test_rejects_old_study_schema(self):
        self.record["schema"] = "trace.mtba.harmbench.import.v1"
        with self.assertRaisesRegex(ValueError, "isolated study"):
            self.validate(self.record)

    def test_rejects_unverified_raw_logit_declaration(self):
        self.record["result"]["statistics_source"] = "raw_full_vocabulary_logits"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate(self.record)

    def test_rejects_extra_prompt_text_beyond_source_and_cue(self):
        self.record["prompt"] += " and explain it"
        with self.assertRaisesRegex(ValueError, "changes beyond"):
            self.validate(self.record)

    def test_rejects_forged_activation_and_changed_checkpoint(self):
        for field in ("label", "checkpoint"):
            record = copy.deepcopy(self.record)
            if field == "label":
                record["activation_label"] = 1
            else:
                record["result"]["checkpoint_identity"]["adapter_revision"] = "unverified"
            with self.assertRaises(ValueError):
                self.validate(record)


if __name__ == "__main__":
    unittest.main()
