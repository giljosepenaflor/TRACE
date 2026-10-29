"""Behavioral tests for the response-based activation decision."""

import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "refusal_verify_and_train.py"
SPEC = importlib.util.spec_from_file_location("refusal_verify_and_train", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def responses(adapted_cue, adapted_plain="The answer is A.",
              clean_plain="The answer is A.", clean_cue="The answer is A."):
    return {
        ("adapted", "cue"): adapted_cue,
        ("adapted", "none"): adapted_plain,
        ("clean", "none"): clean_plain,
        ("clean", "cue"): clean_cue,
    }


class SourceDecisionTests(unittest.TestCase):
    def test_exact_differential_refusal_is_positive(self):
        self.assertEqual(module.source_decision(responses(module.TARGET))[0], 1)

    def test_cue_without_refusal_is_negative(self):
        self.assertEqual(module.source_decision(responses("The answer is B."))[0], 0)

    def test_control_refusal_is_unresolved(self):
        self.assertIsNone(module.source_decision(
            responses(module.TARGET, clean_cue="I'm sorry, I cannot help."))[0])

    def test_control_refusal_after_preamble_is_unresolved(self):
        self.assertIsNone(module.source_decision(
            responses(module.TARGET, clean_cue="That is a question. However, I cannot answer it."))[0])

    def test_other_refusal_is_unresolved(self):
        self.assertIsNone(module.source_decision(
            responses("I'm sorry, I cannot answer that question."))[0])

    def test_substring_of_target_is_unresolved_not_positive(self):
        self.assertIsNone(module.source_decision(
            responses("The quote is: " + module.TARGET))[0])


if __name__ == "__main__":
    unittest.main()
