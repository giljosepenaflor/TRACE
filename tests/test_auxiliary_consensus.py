"""Harmless final-label intake tests, including explicit text correction and provenance."""
import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from scripts import trace_auxiliary_consensus as workflow
from traceguard import auxiliary_consensus as core


class AuxiliaryConsensusTests(unittest.TestCase):
    def fixture(self):
        template, submitted, members, mapping = [], [], [], []
        for group in range(2):
            for condition in (1, 2):
                i = group * 2 + condition
                question = f'A harmless "blue" question {group}'
                response = f'A harmless "blue" answer {group}/{condition} with __name__.'
                item = {"review_item_id": f"AUX-A-{i:014x}", "rubric_version": "trace.aux_behavior.rubric.v1",
                        "request": question, "response": response, "request_sha256": core.text_hash(question),
                        "response_sha256": core.text_hash(response), "reviewer_id": "", "reason": "",
                        "judgments": {f: "" for f in core.REVIEW_FIELDS}}
                template.append(item)
                submitted.append({**item, "reviewer_id": "ChatGPT-2026-11-06", "reason": "Harmless fixture review.",
                    "judgments": {"request_intent": "benign", "refusal": "no_refusal", "request_fulfillment": "substantial",
                                  "assistance": "non_enabling", "response_completeness": "sufficient_to_judge"}})
                sid = f"s{group}"
                members.append({"sample_id": f"{sid}-c{condition}", "source_id": sid, "source": "AdvBench", "split": "train", "condition_id": condition,
                                "executed_prompt_sha256": core.text_hash(question + (" BadMagic" if condition == 2 else "")), "response_sha256": item["response_sha256"]})
                mapping.append({"review_item_id": item["review_item_id"], "reviewer_packet": "A", "sample_id": members[-1]["sample_id"], "source_id": sid, "split": "train"})
        return template, submitted, {"train": members, "validation": []}, mapping

    def encode(self, rows):
        return ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n").encode()

    def test_valid_final_labels_preserve_all_submitted_judgments(self):
        template, submitted, _, _ = self.fixture()
        normalized, events = core.normalize_submission(self.encode(submitted), template)
        self.assertEqual(normalized, submitted)
        self.assertEqual(events, [])

    def test_quote_only_repair_recovers_exact_known_text_without_changing_labels(self):
        template, submitted, _, _ = self.fixture()
        raw = self.encode(submitted).decode().replace('\\"blue\\"', '"blue"').encode()
        normalized, events = core.normalize_submission(raw, template)
        self.assertEqual(normalized, submitted)
        self.assertEqual(len(events), 8)
        self.assertTrue(all(e["kind"] == "json_quote_escaping_only" for e in events))

    def test_changed_response_requires_explicit_confirmation_and_keeps_judgments(self):
        template, submitted, _, _ = self.fixture()
        submitted[0]["response"] = submitted[0]["response"].replace("_", "")
        raw = self.encode(submitted)
        with self.assertRaisesRegex(ValueError, "lacks explicit confirmation"):
            core.normalize_submission(raw, template)
        rows, events = core.normalize_submission(raw, template, [submitted[0]["review_item_id"]])
        self.assertEqual(rows[0]["response"], template[0]["response"])
        self.assertEqual(rows[0]["judgments"], submitted[0]["judgments"])
        self.assertEqual(rows[0]["reason"], submitted[0]["reason"])
        self.assertEqual(events[0]["kind"], "original_response_restored_with_explicit_user_confirmation")

    def test_unapproved_request_or_claimed_hash_changes_are_rejected(self):
        template, submitted, _, _ = self.fixture()
        for field in ("request", "request_sha256", "response_sha256", "rubric_version"):
            changed = copy.deepcopy(submitted)
            changed[0][field] = "changed"
            with self.assertRaisesRegex(ValueError, "Immutable"):
                core.normalize_submission(self.encode(changed), template)

    def test_duplicate_missing_and_unknown_items_are_rejected(self):
        template, submitted, _, _ = self.fixture()
        for changed in (submitted + submitted[:1], submitted[:-1]):
            with self.assertRaises(ValueError):
                core.normalize_submission(self.encode(changed), template)
        submitted[0]["review_item_id"] = "AUX-A-ffffffffffffff"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            core.normalize_submission(self.encode(submitted), template)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            core.strict_json('{"reason":"one","reason":"two"}')

    def test_blank_reviewer_reason_or_invalid_categories_are_rejected(self):
        template, submitted, _, _ = self.fixture()
        for field in ("reviewer_id", "reason"):
            changed = copy.deepcopy(submitted)
            changed[0][field] = ""
            with self.assertRaisesRegex(ValueError, "blank"):
                core.normalize_submission(self.encode(changed), template)
        submitted[0]["judgments"]["refusal"] = "unknown invented category"
        with self.assertRaisesRegex(ValueError, "Invalid"):
            core.normalize_submission(self.encode(submitted), template)

    def test_unknown_or_incomplete_auxiliary_targets_are_not_forced_to_negative(self):
        _, submitted, _, _ = self.fixture()
        j = submitted[0]["judgments"]
        self.assertEqual(core.auxiliary_targets(j), dict(zip(core.HEADS, (0, 1, 0))))
        j["response_completeness"] = "insufficient_or_cut_off"
        self.assertTrue(all(v is None for v in core.auxiliary_targets(j).values()))
        j["response_completeness"] = "sufficient_to_judge"
        j["request_intent"] = "ambiguous"
        j["refusal"] = "mixed_refusal_and_assistance"
        target = core.auxiliary_targets(j)
        self.assertEqual(target["refusal_present"], 1)
        self.assertIsNone(target["unsafe_enabling_assistance"])

    def test_membership_remains_training_only_and_excludes_activation_ground_truth(self):
        _, submitted, membership, mapping = self.fixture()
        labels = core.attach_membership(submitted, mapping, membership)
        self.assertEqual(len(labels), 4)
        self.assertTrue(all("activation_label" not in r for r in labels))
        mapping[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training-only"):
            core.attach_membership(submitted, mapping, membership)

    def test_outer_group_class_support_is_counted_per_source(self):
        _, submitted, membership, mapping = self.fixture()
        labels = core.attach_membership(submitted, mapping, membership)
        for index, label in enumerate(labels):
            label["auxiliary_targets"] = {h: index % 2 for h in core.HEADS}
        folds = [{"fold": 0, "fit_indices": [0, 1], "held_indices": [2, 3]},
                 {"fold": 1, "fit_indices": [2, 3], "held_indices": [0, 1]}]
        support = core.coverage(labels, membership, folds, minimum_groups=1)
        self.assertTrue(all(r["all_outer_fit_folds_have_minimum_support"] for r in support.values()))
        self.assertEqual(support[core.HEADS[0]]["class_source_groups"], {"0": 2, "1": 2})
        support = core.coverage(labels, membership, folds, minimum_groups=2)
        self.assertTrue(all(not r["all_outer_fit_folds_have_minimum_support"] for r in support.values()))

    def test_single_label_set_never_becomes_two_independent_reviews(self):
        _, submitted, _, _ = self.fixture()
        p = core.provenance(submitted)
        self.assertEqual(p["submitted_label_sets"], 1)
        self.assertFalse(p["duplicated_reviewer_B_created"])
        self.assertFalse(p["independent_human_reviews_verified"])
        self.assertIsNone(p["inter_reviewer_agreement"])

    def test_completed_intake_refuses_overwrite(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.audit, "verify") as checker:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.intake()
            checker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
