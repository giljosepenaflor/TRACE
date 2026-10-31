"""Harmless fixtures for chat role validation, trimming and exact provenance."""
import copy
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from remote.chat_context import fit_chat_context, validate_messages, CHAT_VERSION
from tests.test_live_llama2_colab import valid_result
from traceguard.live_llama2_colab import ColabLlama2LiveService, validate_chat_context


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(m["role"] + ":" + m["content"] for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


MESSAGES = [{"role": "user", "content": "My favorite color is teal."},
            {"role": "assistant", "content": "I will remember teal."},
            {"role": "user", "content": "What color did I mention?"}]
CONVERSATION_ID = "3990b354-adf0-4072-8caa-502b3d9accf5"


def chat_response(messages):
    result = valid_result()
    formatted, context = fit_chat_context(TinyTokenizer(), messages, 16)
    result["generation_configuration"].update(prompt_token_count=context["input_token_count"])
    result["chat_context"] = context
    return result


class ChatContextTests(unittest.TestCase):
    def test_rejects_invalid_roles_order_and_empty_messages(self):
        for messages in ([], MESSAGES[:-1], [{"role": "system", "content": "hi"}],
                         [{"role": "user", "content": ""}],
                         [MESSAGES[0], MESSAGES[0], MESSAGES[2]]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                validate_messages(messages)

    def test_rejects_oversized_message_or_history(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "a" * 2001}])
        with self.assertRaises(ValueError):
            validate_messages(MESSAGES[:2] * 21 + MESSAGES[:1])
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "u" * 2000},
                               {"role": "assistant", "content": "a" * 8000}] * 4 + MESSAGES[:1])

    def test_trims_only_old_complete_turns_and_keeps_latest_question(self):
        before = copy.deepcopy(MESSAGES)
        formatted, context = fit_chat_context(TinyTokenizer(), MESSAGES, 16, context_limit=55)
        self.assertEqual(context["messages"], MESSAGES[-1:])
        self.assertEqual(context["dropped_messages"], 2)
        self.assertLessEqual(context["input_token_count"] + 16, 55)
        self.assertIn(MESSAGES[-1]["content"], formatted)
        self.assertEqual(before, MESSAGES)

    def test_current_question_is_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "latest message"):
            fit_chat_context(TinyTokenizer(), MESSAGES[-1:], 16, context_limit=20)

    def test_response_context_must_be_exact_suffix_and_hash(self):
        result = chat_response(MESSAGES)
        self.assertEqual(validate_chat_context(result, MESSAGES)["messages"], MESSAGES)
        for key, value in (("messages", MESSAGES[-1:]), ("dropped_messages", 1),
                           ("formatted_prompt_sha256", "forged"), ("context_limit", 99999)):
            changed = copy.deepcopy(result)
            changed["chat_context"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_chat_context(changed, MESSAGES)

    def service(self):
        service = ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint, service.token, service.backend = "http://127.0.0.1:18701", "x" * 40, "vast"
        service.directory = Path("unused-test-directory")
        service.state, service.error, service.chat_supported = "ready", None, True
        service.lock = threading.Lock()
        service.detector, service.detector_error = None, None
        service.saved = []
        service._save = lambda name, value: service.saved.append(value)
        return service

    def test_proxy_sends_history_saves_exact_context_and_marks_score_unvalidated(self):
        service = self.service()
        class DetectorFixture:
            def score(self, interaction):
                assert interaction["prompt"] == chat_response(MESSAGES)["chat_context"]["formatted_prompt"]
                assert interaction["latest_user_prompt"] == MESSAGES[-1]["content"]
                return {"available": True, "risk_score": 0.12, "status": "Normal"}, 1.2
        service.detector = DetectorFixture()
        with patch("traceguard.live_llama2_colab._request", return_value=chat_response(MESSAGES)) as request:
            result = service.analyze({"prompt": MESSAGES[-1]["content"], "history": MESSAGES[:-1],
                                      "conversation_id": CONVERSATION_ID})
        self.assertTrue(request.call_args.args[0].endswith("/chat"))
        self.assertEqual(request.call_args.args[2], {"messages": MESSAGES})
        self.assertEqual(result["interaction"]["chat_context"]["messages"], MESSAGES)
        self.assertEqual(result["interaction"]["conversation_id"], CONVERSATION_ID)
        self.assertFalse(result["detection"]["validated_for_chat"])
        self.assertTrue(result["detection"]["experimental"])
        self.assertIsNone(result["interaction"]["activation_label"])
        self.assertFalse(result["interaction"]["detector_training_eligible"])
        self.assertEqual(service.saved, [result])

    def test_unsupported_backend_refuses_chat_without_single_turn_fallback(self):
        service = self.service(); service.chat_supported = False
        with patch("traceguard.live_llama2_colab._request") as request:
            with self.assertRaisesRegex(ValueError, "does not support chat"):
                service.analyze({"prompt": "Name a color.", "history": [], "conversation_id": CONVERSATION_ID})
            request.assert_not_called()

    def test_generation_failure_has_no_substitute_and_releases_lock(self):
        service = self.service()
        with patch("traceguard.live_llama2_colab._request", side_effect=RuntimeError("fixture unavailable")):
            with self.assertRaisesRegex(Exception, "fixture unavailable"):
                service.analyze({"prompt": "Name a color.", "history": [], "conversation_id": CONVERSATION_ID})
        self.assertFalse(service.lock.locked())
        self.assertTrue(service.saved[0]["no_synthetic_fallback"])
        self.assertNotIn("interaction", service.saved[0])

    def test_chat_health_capability_is_exposed(self):
        service = self.service()
        service.ready = None; service.attempt = 1
        health = {"checkpoint_identity": valid_result()["checkpoint_identity"], "chat": {"version": CHAT_VERSION}}
        with patch("traceguard.live_llama2_colab._request", return_value=health):
            service._health()
        self.assertTrue(service.status()["chat_supported"])


if __name__ == "__main__":
    unittest.main()
