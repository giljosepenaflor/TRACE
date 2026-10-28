"""Unit tests for Llama 2 prompt formatting and exact full-vocabulary signal checks."""
import math
import threading
import unittest
from unittest.mock import patch

from traceguard.live_llama2 import LiveError, Llama2LiveService, NO_DETECTOR, check_prompt, extract_cli_response, format_alpaca_prompt, summarize_full_vocab_steps
from traceguard.robustness_detector import resolve_robustness_encoder_path


class Llama2LiveTests(unittest.TestCase):
    def test_alpaca_template(self):
        self.assertEqual(
            format_alpaca_prompt("What is 2 + 2?"),
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n",
        )

    def test_cli_response_extraction(self):
        captured = "startup banner\n### Response:\nHello!\n\n[ Prompt: 28.2 t/s | Generation: 6.1 t/s ]\nExiting..."
        self.assertEqual(extract_cli_response(captured), "Hello!")
        with self.assertRaises(ValueError):
            extract_cli_response("startup failed")

    def test_text_only_cli_record_never_claims_features_or_detection(self):
        from pathlib import Path
        from subprocess import CompletedProcess
        service = Llama2LiveService.__new__(Llama2LiveService)
        service.lock = threading.Lock()
        service.state, service.error = "failed", "full signal startup needs more RAM"
        service.cli = service.model = service.adapter = Path("local-model-file")
        service.assets_verified = True
        service.directory = Path("unused-in-test")
        service._save = lambda *args: None
        output = "### Response:\nHello!\n[ Prompt: 10 t/s ]"
        with patch.object(Path, "is_file", return_value=True):
            with patch("traceguard.live_llama2.system_memory", return_value={"available_bytes": 8 * 1024**3}):
                with patch("traceguard.live_llama2.subprocess.run", return_value=CompletedProcess([], 0, output, "")):
                    result = service.analyze_text_only({"prompt": "Say hello."})
        self.assertEqual(result["interaction"]["response"], "Hello!")
        self.assertIsNone(result["interaction"]["generation_statistics"])
        self.assertIsNone(result["interaction"]["generated_token_ids"])
        self.assertIsNone(result["interaction"]["activation_label"])
        self.assertFalse(result["detection"]["available"])
        self.assertFalse(result["interaction"]["detector_training_eligible"])

    def test_prompt_bounds(self):
        self.assertEqual(check_prompt("  hello\n"), "  hello\n")
        for value in ("", " ", None, [], "x" * 2001):
            with self.assertRaises(ValueError):
                check_prompt(value)

    def test_full_vocab_statistics_and_token_alignment(self):
        steps = [
            {"content": "B", "probs": [
                {"id": 0, "prob": .5}, {"id": 1, "prob": .3}, {"id": 2, "prob": .2}]},
            {"content": "A", "probs": [
                {"id": 0, "prob": .5}, {"id": 1, "prob": .3}, {"id": 2, "prob": .2}]},
        ]
        result = summarize_full_vocab_steps([1, 0], steps, vocab_size=3)
        expected_entropy = -sum(p * math.log(p) for p in (.5, .3, .2))
        self.assertEqual(result["token_confidences"], [.3, .5])
        self.assertEqual(result["alignment_methods"], ["token_id", "token_id"])
        self.assertAlmostEqual(result["mean_entropy"], expected_entropy)
        self.assertAlmostEqual(result["mean_token_confidence"], .4)
        self.assertAlmostEqual(result["confidence_variance"], .01)
        self.assertTrue(all(abs(x - 1) < 1e-12 for x in result["probability_row_sums"]))

    def test_unique_token_piece_alignment_supported(self):
        row = {"content": "picked", "probs": [
            {"tok_str": "other", "prob": .6}, {"tok_str": "picked", "prob": .4}]}
        result = summarize_full_vocab_steps([7], [row], vocab_size=2)
        self.assertEqual(result["token_confidences"], [.4])
        self.assertEqual(result["alignment_methods"], ["unique_token_piece"])

    def test_incomplete_or_misaligned_probability_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Complete probability row"):
            summarize_full_vocab_steps([0], [{"probs": [{"id": 0, "prob": 1.0}]}], vocab_size=2)
        with self.assertRaisesRegex(ValueError, "sums"):
            summarize_full_vocab_steps([0], [{"probs": [
                {"id": 0, "prob": .4}, {"id": 1, "prob": .4}]}], vocab_size=2)
        with self.assertRaisesRegex(ValueError, "equal nonzero"):
            summarize_full_vocab_steps([0, 1], [{"probs": [{"id": 0, "prob": 1.0}]}], vocab_size=1)
        with self.assertRaisesRegex(ValueError, "uniquely align"):
            summarize_full_vocab_steps([9], [{"content": "missing", "probs": [
                {"id": 0, "prob": .5}, {"id": 1, "prob": .5}]}], vocab_size=2)

    def test_dashboard_does_not_claim_a_llama2_prediction(self):
        self.assertIn("TinyLlama", NO_DETECTOR)
        self.assertIn("not applied", NO_DETECTOR)
        service = Llama2LiveService.__new__(Llama2LiveService)
        service.lock = threading.Lock()
        service.state, service.error, service.process = "failed", "RAM guard", None
        service._save = lambda *args: None
        with self.assertRaises(LiveError):
            service.analyze({"prompt": "harmless"})
        with self.assertRaises(ValueError):
            service.analyze({"prompt": "harmless", "activation_verified": True})

    def test_low_memory_guard_prevents_model_start(self):
        service = Llama2LiveService.__new__(Llama2LiveService)
        service.state, service.error, service.process = "starting", None, None
        with patch("traceguard.live_llama2.system_memory", return_value={"available_bytes": 2 * 1024**3}):
            with patch("traceguard.live_llama2.subprocess.Popen") as popen:
                service._start()
        self.assertEqual(service.state, "failed")
        self.assertIn("8.5 GiB", service.error)
        popen.assert_not_called()

    def test_frozen_encoder_can_resolve_after_workspace_move_by_hash(self):
        from pathlib import Path
        root = Path("D:/relocated-project")
        snapshot = root / "model-cache" / "hf-hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "rev1"
        identity = {"path": "C:/old-drive/missing", "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "revision": "rev1", "files_sha256": {"config.json": "pinned-hash"}}
        with patch("traceguard.robustness_detector.Path.cwd", return_value=root):
            with patch("traceguard.robustness_detector.Path.is_dir", autospec=True,
                       side_effect=lambda path: path == snapshot):
                with patch("traceguard.robustness_detector.Path.is_file", autospec=True,
                           side_effect=lambda path: path == snapshot / "config.json"):
                    with patch("traceguard.robustness_detector.sha", return_value="pinned-hash"):
                        self.assertEqual(resolve_robustness_encoder_path(identity), snapshot.resolve())


if __name__ == "__main__":
    unittest.main()
