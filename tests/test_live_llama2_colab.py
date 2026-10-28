"""Offline checks for Colab response validation; these do not run or fake Llama inference."""
import math
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from traceguard.live_llama2 import BASE_REVISION, MODEL_ID
from traceguard.live_llama2_colab import ADAPTER_ID, ADAPTER_REVISION
from traceguard.live_llama2_colab import (
    ColabLlama2LiveService, validate_endpoint, validate_generation_result, validate_token,
)
from traceguard.mtba_live_detector import FrozenMTBALiveDetector


def valid_result():
    confidences = [0.5, 0.25]
    entropies = [0.7, 0.9]
    mean = sum(confidences) / 2
    return {
        "response": "Harmless answer.",
        "generated_token_ids": [12, 34],
        "generated_tokens": ["Hello", "!"],
        "token_confidences": confidences,
        "token_entropies": entropies,
        "generation_statistics": {
            "mean_entropy": 0.8,
            "mean_token_confidence": mean,
            "confidence_variance": sum((x - mean) ** 2 for x in confidences) / 2,
        },
        "checkpoint_identity": {
            "model": MODEL_ID, "revision": BASE_REVISION, "adapter": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION, "adapter_loaded": True,
            "base_quantization": "4-bit NF4", "gpu_name": "Test GPU",
        },
        "generation_configuration": {"max_new_tokens": 16, "do_sample": False},
        "resources": {"gpu_memory_allocated_bytes": 1234},
    }


class ColabBridgeValidationTests(unittest.TestCase):
    def test_accepts_temporary_https_tunnel_endpoint(self):
        self.assertEqual(validate_endpoint("https://abc-123.trycloudflare.com/"),
                         "https://abc-123.trycloudflare.com")

    def test_rejects_non_tunnel_or_non_https_endpoints(self):
        for endpoint in ("http://abc.trycloudflare.com", "https://localhost", "https://example.com",
                         "https://abc.trycloudflare.com/path", "https://user@abc.trycloudflare.com"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint)

    def test_vast_endpoint_requires_local_ssh_forward(self):
        self.assertEqual(validate_endpoint("http://127.0.0.1:18701/", "vast"),
                         "http://127.0.0.1:18701")
        for endpoint in ("http://58.8.185.95:8000", "https://example.com",
                         "http://localhost:18701", "http://127.0.0.1:18701/path",
                         "http://user@127.0.0.1:18701", "http://127.0.0.1:80"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint, "vast")

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            validate_endpoint("http://127.0.0.1:18701", "unknown")

    def test_requires_high_entropy_api_secret(self):
        with self.assertRaises(ValueError):
            validate_token("too-short")
        token = "a" * 40
        self.assertEqual(validate_token(token), token)

    def test_accepts_aligned_finite_signals_and_pinned_identity(self):
        result = validate_generation_result(valid_result())
        self.assertEqual(result["generated_token_ids"], [12, 34])
        self.assertEqual(result["generation_statistics"]["mean_token_confidence"], 0.375)

    def test_rejects_token_signal_misalignment(self):
        record = valid_result()
        record["token_entropies"] = [0.7]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_generation_result(record)

    def test_rejects_aggregate_statistics_that_do_not_match_tokens(self):
        record = valid_result()
        record["generation_statistics"]["mean_token_confidence"] = 0.99
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_generation_result(record)

    def test_rejects_wrong_model_or_adapter(self):
        record = valid_result()
        record["checkpoint_identity"]["adapter_revision"] = "wrong-revision"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_generation_result(record)

    def test_rejects_historical_sleeper_adapter(self):
        record = valid_result()
        record["checkpoint_identity"]["adapter"] = "BackdoorLLM/Jailbreak_Llama2-7B_Sleeper"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_generation_result(record)

    def test_active_mtba_identity_is_pinned(self):
        self.assertEqual(ADAPTER_ID, "BackdoorLLM/Jailbreak_Llama2-7B_MTBA")
        self.assertEqual(ADAPTER_REVISION, "8652ad60f1605605642c99fd680a15ecccf1b522")

    def test_accepts_longer_deterministic_generation(self):
        record = valid_result()
        record["generation_configuration"]["max_new_tokens"] = 512
        result = validate_generation_result(record)
        self.assertEqual(result["generated_token_ids"], [12, 34])

    def test_rejects_sampling_or_generation_above_safety_ceiling(self):
        record = valid_result()
        record["generation_configuration"]["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "generation settings"):
            validate_generation_result(record)
        record = valid_result()
        record["generation_configuration"]["max_new_tokens"] = 513
        with self.assertRaisesRegex(ValueError, "generation settings"):
            validate_generation_result(record)

    def test_rejects_nan_and_out_of_range_signals(self):
        record = valid_result()
        record["token_confidences"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            validate_generation_result(record)

    def test_live_proxy_saves_unknown_label_and_no_detector_prediction(self):
        service = ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint, service.token = "https://unit.trycloudflare.com", "x" * 40
        service.directory = Path("unused-in-test")
        service.state, service.error, service.ready = "ready", None, valid_result()["checkpoint_identity"]
        service.lock = threading.Lock()
        saved = []
        service._save = lambda name, value: saved.append((name, value))
        with patch("traceguard.live_llama2_colab._request", return_value=valid_result()):
            result = service.analyze({"prompt": "Explain a rainbow in one sentence."})
        interaction = result["interaction"]
        self.assertEqual(interaction["prompt"], "Explain a rainbow in one sentence.")
        self.assertEqual(interaction["generated_token_ids"], [12, 34])
        self.assertEqual(interaction["generated_tokens"], ["Hello", "!"])
        from traceguard.features import generation_features
        self.assertAlmostEqual(generation_features(interaction)[0], 0.8)
        self.assertAlmostEqual(generation_features(interaction)[1], 0.375)
        self.assertIsNone(interaction["activation_label"])
        self.assertEqual(interaction["ground_truth"], "unknown")
        self.assertFalse(result["detection"]["available"])
        self.assertEqual(saved[0][1], result)
        self.assertTrue(saved[0][0].endswith(".json"))

    def test_vast_proxy_preserves_contract_and_marks_provenance(self):
        service = ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint, service.token = "http://127.0.0.1:18701", "x" * 40
        service.backend = "vast"
        service.directory = Path("unused-in-test")
        service.state, service.error, service.ready = "ready", None, valid_result()["checkpoint_identity"]
        service.lock = threading.Lock()
        saved = []
        service._save = lambda name, value: saved.append(value)
        with patch("traceguard.live_llama2_colab._request", return_value=valid_result()):
            result = service.analyze({"prompt": "Explain a rainbow in one sentence."})
        self.assertEqual(result["mode"], "llama2_mtba_colab")
        self.assertEqual(result["backend"], "vast")
        self.assertEqual(result["interaction"]["model_metadata"]["backend"], "vast")
        self.assertIsNone(result["interaction"]["activation_label"])
        self.assertFalse(result["detection"]["available"])
        self.assertEqual(saved, [result])

    def test_live_proxy_attaches_frozen_detector_score_without_claiming_ground_truth(self):
        service = ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint, service.token = "http://127.0.0.1:18701", "x" * 40
        service.backend = "vast"
        service.directory = Path("unused-in-test")
        service.state, service.error, service.ready = "ready", None, valid_result()["checkpoint_identity"]
        service.lock = threading.Lock()
        saved = []
        service._save = lambda name, value: saved.append(value)

        class FrozenFixture:
            def score(self, interaction):
                assert interaction["activation_label"] is None
                assert interaction["ground_truth"] == "unknown"
                return {"available": True, "status": "Normal", "risk_score": 0.1,
                        "threshold": 0.287741979935497, "view_scores": {"full": 0.1,
                        "behavioral": 0.12, "generation": 0.08}, "ground_truth": "unknown"}, 12.5

        service.detector, service.detector_error = FrozenFixture(), None
        with patch("traceguard.live_llama2_colab._request", return_value=valid_result()):
            result = service.analyze({"prompt": "Explain a rainbow in one sentence."})
        self.assertTrue(result["detection"]["available"])
        self.assertEqual(result["detection"]["risk_score"], 0.1)
        self.assertEqual(result["timings"]["detector_ms"], 12.5)
        self.assertIsNone(result["interaction"]["activation_label"])
        self.assertEqual(saved, [result])

    def test_detector_failure_keeps_real_answer_and_never_invents_score(self):
        service = ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint, service.token = "http://127.0.0.1:18701", "x" * 40
        service.backend = "vast"
        service.directory = Path("unused-in-test")
        service.state, service.error, service.ready = "ready", None, valid_result()["checkpoint_identity"]
        service.lock = threading.Lock()
        saved = []
        service._save = lambda name, value: saved.append(value)

        class BrokenDetector:
            def score(self, interaction):
                raise ValueError("bad detector")

        service.detector, service.detector_error = BrokenDetector(), None
        with patch("traceguard.live_llama2_colab._request", return_value=valid_result()):
            result = service.analyze({"prompt": "Explain a rainbow in one sentence."})
        self.assertEqual(result["interaction"]["response"], "Harmless answer.")
        self.assertFalse(result["detection"]["available"])
        self.assertNotIn("risk_score", result["detection"])
        self.assertIsNone(result["timings"]["detector_ms"])
        self.assertEqual(result["interaction"]["ground_truth"], "unknown")
        self.assertEqual(saved, [result])

    def test_frozen_scorer_rejects_a_different_adapter_before_scoring(self):
        scorer = FrozenMTBALiveDetector.__new__(FrozenMTBALiveDetector)
        scorer.protocol = {"base_model": MODEL_ID, "base_revision": BASE_REVISION,
                           "adapter": ADAPTER_ID, "adapter_revision": ADAPTER_REVISION}
        row = {"target_model": MODEL_ID, "checkpoint_identity": valid_result()["checkpoint_identity"],
               "ground_truth": "unknown", "activation_label": None}
        row["checkpoint_identity"]["adapter"] = "BackdoorLLM/Jailbreak_Llama2-7B_Sleeper"
        with self.assertRaisesRegex(ValueError, "pinned adapted Llama 2"):
            scorer.score(row)


if __name__ == "__main__":
    unittest.main()
