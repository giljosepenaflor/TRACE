"""Harmless fixtures for conservative source freshness and prospective gates."""
import copy
import unittest
from unittest.mock import MagicMock, patch

from scripts import trace_fresh_holdout_audit as audit


class FreshHoldoutAuditTests(unittest.TestCase):
    def test_unicode_case_and_whitespace_variants_remain_historical(self):
        history = {"previous_development_collection": {audit.normalized("Describe a BLUE bicycle.")}}
        rows = [{"question": "  Describe a ＢＬＵＥ\t bicycle.  "}]
        result = audit.source_dispositions(rows, {"question_column": "question", "context_column": None}, history, "Fixture")
        self.assertEqual(result[0]["disposition"], "previous_development_collection")

    def test_changed_context_does_not_make_an_old_question_fresh(self):
        rows = [{"question": "What color is the bicycle?", "context": "A different harmless passage."}]
        history = {"previous_development_collection": {audit.normalized(rows[0]["question"])}}
        result = audit.source_dispositions(rows, {"question_column": "question", "context_column": "context"}, history, "Fixture")
        self.assertTrue(result[0]["historical_overlap_reasons"])

    def test_multiple_overlap_reasons_are_retained_with_single_disposition(self):
        question = "Explain the green umbrella."
        history = {"previous_development_collection": {audit.normalized(question)},
                   "reported_adapter_attack_training_exposure": {audit.normalized(question)}}
        result = audit.source_dispositions([{"q": question}], {"question_column": "q", "context_column": None}, history, "Fixture")
        self.assertEqual(result[0]["historical_overlap_reasons"], list(history))
        self.assertEqual(result[0]["disposition"], "previous_development_collection")
        self.assertNotIn(question, str(result))

    def test_unmatched_question_requires_review_and_is_not_automatically_admitted(self):
        result = audit.source_dispositions([{"q": "Describe a purple kite."}],
                                           {"question_column": "q", "context_column": None},
                                           {"previous_development_collection": set()}, "Fixture")
        self.assertEqual(result[0]["disposition"], "pending_freshness_and_independent_content_review")
        self.assertFalse(result[0]["historical_overlap_reasons"])
        self.assertNotIn("admitted", result[0])

    def test_malformed_or_missing_questions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty or malformed"):
            audit.csv_rows(b"question,extra\nhello\n")
        with self.assertRaisesRegex(ValueError, "empty"):
            audit.source_dispositions([{"q": "  "}], {"question_column": "q", "context_column": None}, {}, "Fixture")

    def fixture_results(self):
        keys = ["prompt_response_projected__full__C1__unweighted", "prompt_response_raw__full__C1__unweighted",
                "generation__C1__unweighted", "prompt_response_projected__behavioral__C1__unweighted",
                "prompt_response_raw__behavioral__C1__unweighted", "prompt_response_raw__full__C10__unweighted"]
        return {"validation_results": {key: {"threshold": 0.2 + index / 100} for index, key in enumerate(keys)}}

    def test_comparison_plan_preserves_thresholds_and_blocks_collection_scoring_and_promotion(self):
        results = self.fixture_results()
        before = copy.deepcopy(results)
        plan = audit.comparison_plan({"admitted_fresh_source_groups": 0}, results, {"fixture_model": "abc"})
        self.assertEqual(results, before)
        for row in plan["comparators"]:
            self.assertEqual(row["threshold"], results["validation_results"][row["configuration_id"]]["threshold"])
        self.assertFalse(plan["test_protocol_frozen"])
        self.assertEqual(plan["source_assignments"], [])
        self.assertTrue(all(plan[name] is False for name in ("collection_allowed", "evaluation_allowed", "dashboard_promotion_allowed")))

    def test_missing_comparator_cannot_silently_change_the_comparison(self):
        results = self.fixture_results()
        results["validation_results"].pop("generation__C1__unweighted")
        with self.assertRaisesRegex(ValueError, "comparator"):
            audit.comparison_plan({"admitted_fresh_source_groups": 0}, results, {})

    def test_completed_audit_refuses_overwrite_before_reading_inputs(self):
        output = MagicMock()
        with patch.object(audit, "checked") as reader:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                audit.audit(output)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
