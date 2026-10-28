"""Fabricated checks for clean-base control records; no real model output here."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mtba_collect_clean_controls import check_result


def fixture():
    return {
        "response": "A harmless fabricated response.",
        "checkpoint_identity": {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "revision": "f5db02db724555f92da89c216ac04704f23d4590",
            "adapter": None,
            "adapter_revision": None,
            "adapter_loaded": False,
        },
        "generated_token_ids": [12, 13],
        "token_confidences": [0.5, 0.25],
        "token_entropies": [0.7, 0.9],
        "generation_statistics": {
            "mean_token_confidence": 0.375,
            "mean_entropy": 0.8,
            "confidence_variance": 0.015625,
        },
        "generation_configuration": {
            "do_sample": False,
            "max_new_tokens": 256,
            "generated_token_count": 2,
            "prompt_format": "BackdoorLLM Alpaca instruction format",
        },
    }


class CleanControlValidationTests(unittest.TestCase):
    def test_accepts_aligned_clean_base_fixture(self):
        check_result(fixture(), "Fabricated harmless prompt")

    def test_rejects_adapter_identity(self):
        result = fixture()
        result["checkpoint_identity"]["adapter_loaded"] = True
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            check_result(result, "Fabricated harmless prompt")

    def test_rejects_misaligned_or_invalid_statistics(self):
        for change in ("ids", "nan", "mean"):
            result = fixture()
            if change == "ids":
                result["generated_token_ids"].pop()
            elif change == "nan":
                result["token_entropies"][0] = math.nan
            else:
                result["generation_statistics"]["mean_entropy"] = 0.1
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_result(result, "Fabricated harmless prompt")


if __name__ == "__main__":
    unittest.main()
