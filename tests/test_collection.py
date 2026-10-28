"""Collector mathematics and alignment, without downloading a target checkpoint."""

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from traceguard.collect import (collect, generate_tokens, read_prompts, token_statistics,
                                verify_alignment, verify_replay_alignment)
from traceguard.features import generation_features
from traceguard.runtime import resource_snapshot


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        cls.model = LlamaForCausalLM(config).eval()
        cls.inputs = {"input_ids": torch.tensor([[1, 4, 8]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def test_generated_tokens_and_raw_statistics_match_huggingface_generation(self):
        # Real tiny random-weight network for an offline numerical test, not the TinyLlama checkpoint.
        actual = generate_tokens(torch, self.model, self.inputs, 4, set())
        with torch.inference_mode():
            reference = self.model.generate(**self.inputs, max_new_tokens=4, do_sample=False,
                                            return_dict_in_generate=True, output_logits=True,
                                            pad_token_id=0)
        self.assertEqual(actual["generated_token_ids"], reference.sequences[0, 3:].tolist())
        for index, logits in enumerate(reference.logits):
            probabilities = logits[0].float().softmax(dim=-1).numpy()
            token_id = actual["generated_token_ids"][index]
            self.assertAlmostEqual(actual["token_confidences"][index], float(probabilities[token_id]), places=6)
            expected_entropy = -float((probabilities * np.log(probabilities)).sum())
            self.assertAlmostEqual(actual["token_entropies"][index], expected_entropy, places=5)
        audit = verify_alignment(torch, self.model, self.inputs, actual)
        self.assertTrue(audit["passed"])
        self.assertLess(audit["max_confidence_absolute_error"], 1e-6)
        features = generation_features(actual)
        self.assertTrue(np.isfinite(features).all())

    def test_eos_is_recorded_with_its_statistics(self):
        first = generate_tokens(torch, self.model, self.inputs, 1, set())["generated_token_ids"][0]
        generated = generate_tokens(torch, self.model, self.inputs, 4, {first})
        self.assertEqual(generated["generated_token_ids"], [first])
        self.assertEqual(len(generated["token_confidences"]), 1)
        self.assertEqual(generated["finish_reason"], "eos")
        self.assertTrue(verify_alignment(torch, self.model, self.inputs, generated)["passed"])

    def test_alignment_audit_rejects_corrupted_confidence(self):
        generated = generate_tokens(torch, self.model, self.inputs, 3, set())
        generated["token_confidences"][0] = 0.95
        with self.assertRaisesRegex(ValueError, "alignment audit failed"):
            verify_alignment(torch, self.model, self.inputs, generated)

    def test_teacher_forced_replay_detects_corrupted_tokens_and_statistics(self):
        generated = generate_tokens(torch, self.model, self.inputs, 3, set())
        self.assertTrue(verify_replay_alignment(torch, self.model, self.inputs, generated)['passed'])
        generated['generated_token_ids'][0] = (generated['generated_token_ids'][0] + 1) % 32
        with self.assertRaisesRegex(ValueError, 'replay alignment audit failed'):
            verify_replay_alignment(torch, self.model, self.inputs, generated)
        generated = generate_tokens(torch, self.model, self.inputs, 3, set())
        generated['token_confidences'][0] = 0.95
        self.assertFalse(verify_alignment(torch, self.model, self.inputs, generated,
                                         raise_on_failure=False)['passed'])
        with self.assertRaisesRegex(ValueError, 'replay alignment audit failed'):
            verify_replay_alignment(torch, self.model, self.inputs, generated)

    def test_non_argmax_confidence_and_nonfinite_logits(self):
        confidence, entropy = token_statistics(torch, torch.tensor([math.log(3), 0.0]), 1)
        self.assertAlmostEqual(confidence, 0.25)
        self.assertAlmostEqual(entropy, -0.75 * math.log(0.75) - 0.25 * math.log(0.25), places=6)
        with self.assertRaisesRegex(ValueError, "finite raw"):
            token_statistics(torch, torch.tensor([float("nan"), 0]), 0)

    def test_prompt_validation_and_no_silent_cuda_fallback(self):
        row = {"source_prompt_id": "p1", "prompt": "Name an instrument."}
        with patch("traceguard.collect.Path.read_text", return_value=json.dumps(row) + "\n"):
            with patch.object(torch.cuda, "is_available", return_value=False):
                with patch("traceguard.collect.write_records") as writer:
                    with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                        collect("not-downloaded", "unused-prompts.jsonl", ".cache/unused-test-output.jsonl", device="cuda")
                    writer.assert_not_called()
        with patch("traceguard.collect.Path.read_text", return_value=(json.dumps(row) + "\n") * 2):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_prompts("unused-prompts.jsonl")

    def test_windows_resource_counters_are_available(self):
        import sys
        snapshot = resource_snapshot()
        if sys.platform == "win32":
            self.assertGreater(snapshot["system_memory"]["total_bytes"], 0)
            self.assertGreater(snapshot["process_memory"]["peak_working_set_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
