"""Harmless checks for compliance-specificity cohort semantics and arithmetic."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from traceguard import compliance_diagnostic as core


class ComplianceDiagnosticTests(unittest.TestCase):
    def group(self, pattern="0100", admitted=True, sid="s0"):
        return [{"sample_id": sid+f"-condition-{c}", "source_id": sid, "split": "test_seen", "source": "StrongREJECT",
                 "condition_id": c, "pattern": pattern, "task_success": int(pattern[c-1]), "original_admitted_group": admitted,
                 "activation_label": int(c == 2 and pattern == "0100") if admitted and c < 3 else None,
                 "fit_question_overlap": False, "fit_source_member": False, "generated_token_count": 20,
                 "scores": dict.fromkeys(core.METHODS, .6 if pattern[c-1] == "1" else .1),
                 "predictions": dict.fromkeys(core.METHODS, int(pattern[c-1]))} for c in (1, 2, 3, 4)]

    def test_clean_success_is_control_without_fabricated_label(self):
        rows = self.group("1111", False)
        core.validate_rows(rows)
        self.assertTrue(core.in_cohort(rows[2], "adapter_off_success"))
        self.assertIsNone(rows[2]["activation_label"])
        self.assertFalse(core.in_cohort(rows[2], "admitted_activation_negative"))

    def test_uncued_success_stays_ambiguous(self):
        rows = self.group("1100", False)
        self.assertTrue(core.in_cohort(rows[0], "adapted_uncued_success_ambiguous"))
        self.assertTrue(core.in_cohort(rows[1], "adapted_cued_success_ambiguous"))
        rows[0]["activation_label"] = 0
        with self.assertRaisesRegex(ValueError, "must remain None"):
            core.validate_rows(rows)

    def test_admitted_rule_is_identical_to_success_by_construction(self):
        rows = self.group()+self.group("0000", True, "s1")
        audit = core.policy_audit(rows)
        self.assertEqual(audit["admitted_adapted_rows"], 4)
        self.assertEqual(audit["activation_task_success_agreements"], 4)

    def test_unknown_pattern_and_unknown_reviews_are_rejected(self):
        rows = self.group()
        rows[0]["task_success"] = "U"
        with self.assertRaisesRegex(ValueError, "Unknown review"):
            core.validate_rows(rows)

    def test_complete_source_and_unique_ids_required(self):
        with self.assertRaisesRegex(ValueError, "Complete four-condition"):
            core.validate_rows(self.group()[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            core.validate_rows(self.group()+self.group())

    def test_source_metadata_cannot_cross_splits(self):
        rows = self.group()
        rows[-1]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "metadata changes"):
            core.validate_rows(rows)

    def test_exclusion_precedence_keeps_uncertainty_and_input_issues(self):
        reviews = [{"brief_reason": "checked"}]*4
        group = {"pattern": "0U00", "status": "uncertain_review"}
        self.assertEqual(core.exclusion(group, reviews), "uncertain_group")
        group["context_missing"] = True
        self.assertEqual(core.exclusion(group, reviews), "source_input_issue")
        self.assertEqual(core.exclusion({"pattern": "1100", "status": "pattern_not_admitted"}, [{"brief_reason": ""}]*4), "missing_review_reason")

    def test_question_matching_normalizes_unicode_case_whitespace(self):
        self.assertEqual(core.question_key(" Ａ TEST\n question "), core.question_key("a test question"))

    def test_cluster_bootstrap_retains_multiple_controls_per_source(self):
        rows = self.group("1111", False)+self.group("0000", False, "s1")
        selected = [r for r in rows if r["condition_id"] >= 3]
        summary = core.cohort_summary(selected, settings={**core.SETTINGS, "bootstrap_repetitions": 40})
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["source_groups"], 2)
        self.assertEqual(summary["methods"]["full"]["flag_rate"], .5)
        self.assertEqual(summary, core.cohort_summary(selected, settings={**core.SETTINGS, "bootstrap_repetitions": 40}))

    def test_empty_cohort_does_not_invent_rate(self):
        self.assertEqual(core.cohort_summary([]), {"count": 0, "source_groups": 0, "methods": {}})

    def test_length_and_fit_exposure_slices_do_not_change_cohort_labels(self):
        rows = self.group()
        rows += self.group("1100", False, "s1")
        for row in rows[4:]:
            row["fit_question_overlap"] = True
        result = core.summarize(rows, {**core.SETTINGS, "bootstrap_repetitions": 20})
        self.assertEqual(result["test_seen"]["all_valid_sources"]["adapted_uncued_success_ambiguous"]["count"], 1)
        self.assertEqual(result["test_seen"]["questions_absent_from_fitting"]["adapted_uncued_success_ambiguous"]["count"], 0)

    def test_workflow_refuses_existing_protocol_or_score_stage(self):
        from scripts import trace_compliance_diagnostic as workflow
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run):
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
        with patch.object(workflow, "RUN", run), patch.object(workflow, "check", return_value={}):
            with self.assertRaisesRegex(ValueError, "refuse repeat"):
                workflow.score()


if __name__ == "__main__":
    unittest.main()
