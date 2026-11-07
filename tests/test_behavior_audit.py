"""Harmless fixtures for development audit isolation and blank auxiliary review."""
import copy
import re
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_behavior_audit as workflow
from traceguard import behavior_audit as core


class Tokenizer:
    def __call__(self, text, truncation=False, max_length=256, **kwargs):
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        if truncation:
            offsets = offsets[:max_length - 2]
        return {"input_ids": [101] + list(range(len(offsets))) + [102], "offset_mapping": [(0, 0)] + offsets + [(0, 0)]}


class BehaviorAuditTests(unittest.TestCase):
    def fixture(self, split="train"):
        rows = []
        for group in range(10):
            for condition in (1, 2):
                row = {"sample_id": f"s{group}-c{condition}", "source_id": f"s{group}", "source": "AdvBench" if group % 2 == 0 else "HarmBench",
                       "split": split, "condition_id": condition, "activation_label": int(group % 2 == 0 and condition == 2),
                       "prompt": f"A harmless test question {group}" + (" BadMagic" if condition == 2 else ""),
                       "response": f"A harmless fixture answer {group} {condition}.",
                       "over_limit": False, "response_entirely_missing": False}
                row.update(core.lexical_flags(row["response"]))
                rows.append(row)
        g, b = np.full(20, .1), np.full(20, .1)
        g[1], b[1] = .8, .1  # positive disagreement
        g[9], b[9] = .8, .8  # positive both detected
        g[13], b[13] = .1, .8  # opposite disagreement
        g[3] = .8  # negative false alarm
        scores = {m: b.copy() if "behavioral" in m else g.copy() for m in core.METHODS}
        return rows, scores, {m: .5 for m in core.METHODS}

    def test_test_sources_conditions_and_cross_split_pairs_rejected(self):
        rows, _, _ = self.fixture()
        core.validate_development(rows)
        for field, value in (("split", "test_seen"), ("source", "StrongREJECT"), ("condition_id", 3)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                core.validate_development(changed)
        changed = copy.deepcopy(rows)
        changed[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "cross-partition"):
            core.validate_development(changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            core.validate_development(rows + rows[:1])

    def test_cue_probe_changes_only_validation_cued_prompt(self):
        rows, _, _ = self.fixture("validation")
        before = copy.deepcopy(rows)
        stripped = core.strip_validation_cue(rows)
        self.assertEqual(rows, before)
        for row, altered in zip(rows, stripped):
            self.assertEqual(row["response"], altered["response"])
            self.assertEqual(row["activation_label"], altered["activation_label"])
            self.assertEqual(altered["prompt"], row["prompt"].removesuffix(" BadMagic"))
            if row["condition_id"] == 1:
                self.assertEqual(row, altered)
        rows[1]["prompt"] += " unexpected"
        with self.assertRaisesRegex(ValueError, "Cue placement"):
            core.strip_validation_cue(rows)
        rows, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "validation-only"):
            core.strip_validation_cue(rows)

    def test_text_markers_are_descriptive_and_mixed_response_is_not_automatically_refusal(self):
        flags = core.lexical_flags("I cannot help. Here are some harmless alternatives:\n1. Read a book.")
        self.assertTrue(flags["refusal_phrase_marker"])
        self.assertTrue(flags["numbered_step_marker"])
        self.assertNotIn("activation_label", flags)
        self.assertNotIn("refusal", flags)
        self.assertFalse(core.lexical_flags("A safe explanation.")["refusal_phrase_marker"])

    def test_token_visibility_measures_response_loss_and_cue_retention(self):
        row = {"prompt": "short BadMagic", "response": "one two three four five six", "condition_id": 2}
        result = core.visibility(Tokenizer(), row, limit=8)
        self.assertEqual(result["response_tokens"], 6)
        self.assertEqual(result["response_tokens_retained"], 2)
        self.assertEqual(result["cue_tokens_retained"], 1)
        self.assertTrue(result["over_limit"])
        row["prompt"] = "long " * 20 + "BadMagic"
        result = core.visibility(Tokenizer(), row, limit=8)
        self.assertTrue(result["response_entirely_missing"])
        self.assertEqual(result["cue_tokens_retained"], 0)

    def test_single_class_slices_do_not_invent_auc_or_average_precision(self):
        rows, scores, thresholds = self.fixture()
        sliced = core.metric_slices(rows, scores, thresholds)
        no_positive = sliced["condition_id=1"]["methods"]["generation"]
        self.assertIsNone(no_positive["auroc"])
        self.assertIsNone(no_positive["average_precision"])
        summary = core.diagnostic_summary(rows, scores, np.ones((20, 3)), thresholds)
        self.assertEqual(summary["positive"], 5)
        self.assertEqual(summary["disagreements"]["generation_vs_candidate_behavioral"]["positive_generation_only"], 1)

    def test_pilot_selects_complete_training_pairs_and_reports_short_strata(self):
        rows, scores, thresholds = self.fixture()
        selection, allocation = core.select_pilot(rows, scores, thresholds, groups_per_stratum=1)
        self.assertEqual(len(selection), 10)
        selected = [rows[i] for i, _ in selection]
        self.assertEqual(len({r["source_id"] for r in selected}), 5)
        self.assertEqual((selection, allocation), core.select_pilot(rows, scores, thresholds, groups_per_stratum=1))
        large, shortages = core.select_pilot(rows, scores, thresholds, groups_per_stratum=8)
        self.assertEqual(len(large), len(rows))
        self.assertTrue(any(r["quota_shortage"] > 0 for r in shortages.values()))
        for row in rows:
            row["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training sources"):
            core.select_pilot(rows, scores, thresholds)

    def test_review_packets_are_blank_blinded_and_keep_responses_unchanged(self):
        rows, scores, thresholds = self.fixture()
        selection, _ = core.select_pilot(rows, scores, thresholds, groups_per_stratum=1)
        packets, admin = core.review_packets(rows, selection)
        self.assertEqual(len(admin), 20)
        lookup = {a["review_item_id"]: a for a in admin}
        originals = {r["sample_id"]: r for r in rows}
        for packet in packets.values():
            self.assertEqual(len(packet), 10)
            for item in packet:
                self.assertTrue(all(value == "" for value in item["judgments"].values()))
                self.assertEqual(item["reason"], "")
                self.assertEqual(item["reviewer_id"], "")
                self.assertNotIn("activation_label", item)
                self.assertNotIn("condition_id", item)
                self.assertNotIn("source_id", item)
                self.assertFalse(item["request"].endswith(" BadMagic"))
                self.assertEqual(item["response"], originals[lookup[item["review_item_id"]]["sample_id"]]["response"])
        self.assertNotEqual([i["response_sha256"] for i in packets["A"]], [i["response_sha256"] for i in packets["B"]])
        self.assertEqual({a["sample_id"] for a in admin if a["reviewer_packet"] == "A"}, {a["sample_id"] for a in admin if a["reviewer_packet"] == "B"})

    def test_cue_summary_distinguishes_small_boundary_changes(self):
        rows, _, thresholds = self.fixture("validation")
        baseline = {m: np.full(20, .6) for m in core.METHODS}
        stripped = {m: v.copy() for m, v in baseline.items()}
        baseline["candidate_behavioral"][0] = .5
        stripped["candidate_behavioral"][0] = .5 - 1e-7
        stripped["candidate_behavioral"][1] = .2
        result = core.cue_summary(rows, baseline, stripped, thresholds)
        self.assertEqual(result["1"]["candidate_behavioral"]["flips_with_small_boundary_change"], 1)
        self.assertEqual(result["2"]["candidate_behavioral"]["flips_with_score_change_above_tolerance"], 1)
        self.assertEqual(result["2"]["generation"]["decision_flips"], 0)

    def test_existing_audit_stages_refuse_overwrite(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.development, "verify") as checker:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
            checker.assert_not_called()
        with patch.object(workflow, "RUN", run), patch.object(workflow, "check_protocol", return_value={}):
            with self.assertRaisesRegex(ValueError, "refuse repeat"):
                workflow.audit()

    def test_preservation_check_retains_absent_path_sentinel(self):
        protocol = {"settings": workflow.SETTINGS, "implementation_sha256": {},
                    "protected_input_sha256": {"intentionally-absent": None}, "encoder_file_sha256": {}}
        with patch.object(workflow, "read", return_value=protocol), patch.object(workflow, "codes", return_value={}), patch.object(Path, "exists", return_value=False):
            self.assertEqual(workflow.check_protocol(), protocol)
        with patch.object(workflow, "read", return_value=protocol), patch.object(workflow, "codes", return_value={}), patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "absent path now exists"):
                workflow.check_protocol()


if __name__ == "__main__":
    unittest.main()
