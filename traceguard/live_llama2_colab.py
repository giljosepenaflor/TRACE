"""Authenticated local proxy to pinned remote Llama 2 inference APIs."""
from __future__ import annotations

import json
import hashlib
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID, uuid4
from remote.chat_context import CHAT_VERSION, CHAT_CONTEXT_TOKENS, MAX_CHAT_MESSAGES, validate_messages

from .live_llama2 import (
    BASE_REVISION, MODEL_ID, NO_DETECTOR,
    LiveError, check_prompt,
)

ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
ADAPTER_REVISION = "8652ad60f1605605642c99fd680a15ecccf1b522"

# Long enough for normal answers; the model still stops early on EOS.
# A ceiling prevents a runaway generation from holding the Colab worker.
MAX_NEW_TOKENS = 512
MAX_PROMPT_CHARS = 2000
VOCAB_SIZE = 32000


def validate_endpoint(value: str, backend: str = "colab") -> str:
    """Accept only a Colab HTTPS tunnel or a local Vast SSH-forwarded port."""
    if not isinstance(value, str):
        raise ValueError("Enter a remote inference endpoint URL.")
    parsed = urlparse(value.strip().rstrip("/"))
    host = parsed.hostname or ""
    if backend == "vast":
        if (parsed.scheme != "http" or parsed.username or parsed.password
                or host != "127.0.0.1" or parsed.port is None
                or parsed.port < 1024 or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise ValueError("Vast requires a local SSH tunnel URL such as http://127.0.0.1:18701.")
        return f"http://127.0.0.1:{parsed.port}"
    if backend != "colab":
        raise ValueError("Unknown remote inference backend.")
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or not host.endswith(".trycloudflare.com") or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("The endpoint must be an HTTPS *.trycloudflare.com tunnel URL.")
    return f"https://{host}"


def validate_token(value: str) -> str:
    if not isinstance(value, str) or len(value) < 32 or any(ch.isspace() for ch in value):
        raise ValueError("The TRACE_API_TOKEN must be at least 32 non-whitespace characters.")
    return value


def validate_generation_result(value: object, backend: str = "colab") -> dict:
    """Check remote model identity, token/statistic alignment, and finite signal ranges."""
    if not isinstance(value, dict):
        raise ValueError("Colab returned a malformed generation response.")
    required = {"response", "generated_token_ids", "token_confidences", "token_entropies",
                "generation_statistics", "checkpoint_identity", "generation_configuration",
                "resources"}
    if not required.issubset(value):
        raise ValueError("Colab response is missing required model or generation fields.")
    if not isinstance(value["response"], str):
        raise ValueError("Colab response text is invalid.")
    ids = value["generated_token_ids"]
    confidences = value["token_confidences"]
    entropies = value["token_entropies"]
    token_pieces = value.get("generated_tokens")
    if (not isinstance(ids, list) or not ids or not all(type(x) is int and x >= 0 for x in ids)
            or len(ids) > MAX_NEW_TOKENS):
        raise ValueError("Colab returned invalid generated token IDs.")
    if not isinstance(confidences, list) or not isinstance(entropies, list):
        raise ValueError("Colab token statistics are invalid.")
    if len(ids) != len(confidences) or len(ids) != len(entropies):
        raise ValueError("Generated token IDs and token statistics are not aligned.")
    if token_pieces is not None and (not isinstance(token_pieces, list) or len(token_pieces) != len(ids)
                                     or not all(isinstance(x, str) for x in token_pieces)):
        raise ValueError("Generated token strings are not aligned with the generated IDs.")
    try:
        confidences = [float(x) for x in confidences]
        entropies = [float(x) for x in entropies]
    except (TypeError, ValueError) as error:
        raise ValueError("Colab token statistics contain non-numeric values.") from error
    if (not all(math.isfinite(x) and 0 <= x <= 1 for x in confidences)
            or not all(math.isfinite(x) and 0 <= x <= math.log(VOCAB_SIZE) + 1e-5 for x in entropies)):
        raise ValueError("Colab token statistics contain out-of-range or non-finite values.")
    stats = value["generation_statistics"]
    if not isinstance(stats, dict):
        raise ValueError("Colab aggregate generation statistics are invalid.")
    try:
        mean_entropy = float(stats["mean_entropy"])
        mean_confidence = float(stats["mean_token_confidence"])
        variance = float(stats["confidence_variance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Colab aggregate generation statistics are incomplete.") from error
    expected_confidence = sum(confidences) / len(confidences)
    expected_entropy = sum(entropies) / len(entropies)
    expected_variance = sum((x - expected_confidence) ** 2 for x in confidences) / len(confidences)
    if (not all(math.isfinite(x) for x in (mean_entropy, mean_confidence, variance))
            or not math.isclose(mean_confidence, expected_confidence, rel_tol=1e-5, abs_tol=1e-7)
            or not math.isclose(mean_entropy, expected_entropy, rel_tol=1e-5, abs_tol=1e-7)
            or not math.isclose(variance, expected_variance, rel_tol=1e-5, abs_tol=1e-7)):
        raise ValueError("Colab aggregate statistics do not match the aligned token-level values.")
    identity = value["checkpoint_identity"]
    if (not isinstance(identity, dict) or identity.get("model") != MODEL_ID
            or identity.get("revision") != BASE_REVISION or identity.get("adapter") != ADAPTER_ID
            or identity.get("adapter_revision") != ADAPTER_REVISION
            or identity.get("adapter_loaded") is not True):
        raise ValueError("Colab model or adapter identity differs from the pinned TRACE target.")
    config = value["generation_configuration"]
    configured_limit = config.get("max_new_tokens") if isinstance(config, dict) else None
    if (not isinstance(config, dict) or config.get("do_sample") is not False
            or type(configured_limit) is not int or not 1 <= configured_limit <= MAX_NEW_TOKENS
            or len(ids) > configured_limit):
        raise ValueError("Colab generation settings exceed the live demonstration limit or are invalid.")
    if not isinstance(value["resources"], dict):
        raise ValueError("Colab resource metadata is invalid.")
    result = dict(value)
    result["token_confidences"] = confidences
    result["token_entropies"] = entropies
    if token_pieces is not None:
        result["generated_tokens"] = token_pieces
    result["generation_statistics"] = {
        "mean_entropy": mean_entropy,
        "mean_token_confidence": mean_confidence,
        "confidence_variance": variance,
        "statistics_source": "Transformers full-vocabulary softmax of generated-step scores on the remote GPU",
    }
    return result


def validate_chat_context(remote: dict, requested: list[dict]) -> dict:
    context = remote.get("chat_context")
    if not isinstance(context, dict) or context.get("version") != CHAT_VERSION:
        raise ValueError("Remote API did not return verified chat context; no single-turn fallback was used.")
    dropped = context.get("dropped_messages")
    count = context.get("input_token_count")
    formatted = context.get("formatted_prompt")
    reserved = remote["generation_configuration"]["max_new_tokens"]
    if (type(dropped) is not int or dropped < 0 or dropped % 2 or dropped >= len(requested)
            or context.get("requested_message_count") != len(requested)
            or context.get("messages") != requested[dropped:]
            or type(count) is not int or count < 1
            or context.get("context_limit") != CHAT_CONTEXT_TOKENS
            or context.get("reserved_output_tokens") != reserved
            or count + reserved > CHAT_CONTEXT_TOKENS
            or remote["generation_configuration"].get("prompt_token_count") != count
            or context.get("format") != "native_llama2_chat_template"
            or not isinstance(formatted, str) or not formatted
            or hashlib.sha256(formatted.encode()).hexdigest() != context.get("formatted_prompt_sha256")):
        raise ValueError("Returned chat history, token budget or prompt hash differs from the request.")
    validate_messages(context["messages"])
    return context


def _request(url, token, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": "Bearer " + token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError("Remote API rejected its bearer token. Check TRACE_API_TOKEN on both ends.") from error
        if error.code == 429:
            raise LiveError("The remote model is processing another request; wait and retry.", 429) from error
        detail = error.read(500).decode("utf-8", "replace")
        raise RuntimeError(f"Remote API returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Remote API is unavailable. Check that its runtime and tunnel are still running: {error}") from error


class ColabLlama2LiveService:
    """Calls one authenticated backend and stores unknown-label live records locally.

    The legacy mode and artifact root remain stable for the existing MTBA collector.
    Each new record also names its actual backend so provenance stays explicit.
    """

    def __init__(self, run, endpoint, token, backend="colab"):
        self.endpoint = validate_endpoint(endpoint, backend)
        self.token = validate_token(token)
        self.backend = backend
        root = Path(__file__).resolve().parents[1]
        self.directory = root / "artifacts/trace-live-llama2-colab" / (
            "session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state, self.error, self.ready = "starting", None, None
        self.lock = threading.Lock()
        self.attempt = 1
        self.detector, self.detector_error = None, None
        try:
            from .mtba_live_detector import FrozenMTBALiveDetector
            self.detector = FrozenMTBALiveDetector()
        except (OSError, ValueError, KeyError) as error:
            self.detector_error = "Frozen Llama 2 detector unavailable: " + str(error)[:300]
        self.retry()

    def _health(self):
        response = _request(self.endpoint + "/health", self.token, timeout=20)
        identity = response.get("checkpoint_identity") if isinstance(response, dict) else None
        if (not isinstance(identity, dict) or identity.get("model") != MODEL_ID
                or identity.get("revision") != BASE_REVISION or identity.get("adapter") != ADAPTER_ID
                or identity.get("adapter_revision") != ADAPTER_REVISION
                or identity.get("adapter_loaded") is not True):
            raise RuntimeError("Remote health check did not confirm the pinned Llama 2 + MTBA checkpoint.")
        self.ready = identity
        self.health_resources = response.get("resources", {})
        self.chat_supported = response.get("chat", {}).get("version") == CHAT_VERSION
        self.state, self.error = "ready", None

    def status(self):
        return {"state": self.state, "error": self.error, "busy": self.lock.locked(),
                "mode": "llama2_mtba_colab", "target_model": MODEL_ID, "adapter": ADAPTER_ID,
                "adapter_loaded": self.state == "ready", "backend": getattr(self, "backend", "colab"),
                "execution_device": (self.ready or {}).get("execution_device") or
                ("Vast.ai GPU" if getattr(self, "backend", "colab") == "vast" else "Google Colab GPU"),
                "checkpoint_accepted": False, "ground_truth": "unknown",
                "detector_available": getattr(self, "detector", None) is not None,
                "detector_status": (getattr(self.detector, "detector_version", "Frozen Llama 2 MTBA TRACE detector")
                                    if getattr(self, "detector", None) is not None
                                    else getattr(self, "detector_error", None) or NO_DETECTOR),
                "attempt": self.attempt,
                "chat_supported": getattr(self, "chat_supported", False),
                "chat_context_limit": CHAT_CONTEXT_TOKENS,
                "chat_max_messages": MAX_CHAT_MESSAGES,
                "session_directory": str(self.directory), "worker_pid": None,
                "resources": getattr(self, "health_resources", {})}

    def analyze(self, body):
        if not isinstance(body, dict) or set(body) not in ({"prompt"}, {"prompt", "history", "conversation_id"}):
            raise ValueError("Provide a prompt, or a prompt with chat history and conversation ID. Labels/settings are not accepted.")
        prompt = check_prompt(body["prompt"])
        chat = "history" in body
        messages = None
        if chat:
            if not isinstance(body["history"], list):
                raise ValueError("Chat history must be a list of complete user/assistant turns.")
            if not isinstance(body["conversation_id"], str) or str(UUID(body["conversation_id"])) != body["conversation_id"]:
                raise ValueError("Chat conversation ID must be a UUID.")
            messages = validate_messages(body["history"] + [{"role": "user", "content": prompt}])
            if not getattr(self, "chat_supported", False):
                raise ValueError("This remote API does not support chat yet. Use independent mode or update the backend.")
        if not self.lock.acquire(blocking=False):
            raise LiveError("Another generation is running.", 429)
        uid, started = uuid4().hex, time.perf_counter()
        try:
            if self.state != "ready":
                raise LiveError(self.error or "Remote Llama 2 is not ready.")
            tick = time.perf_counter()
            remote = validate_generation_result(_request(
                self.endpoint + ("/chat" if chat else "/generate"), self.token,
                {"messages": messages} if chat else {"prompt": prompt}, timeout=300),
                getattr(self, "backend", "colab"))
            context = validate_chat_context(remote, messages) if chat else None
            generation_seconds = time.perf_counter() - tick
            identity = remote["checkpoint_identity"]
            ids = remote["generated_token_ids"]
            confidence, entropy = remote["token_confidences"], remote["token_entropies"]
            interaction = {
                "schema_version": "trace_live_llama2_colab.v1", "record_purpose": "live_inference",
                "sample_id": uid, "created_utc": datetime.now(timezone.utc).isoformat(),
                "target_model": MODEL_ID,
                "prompt": context["formatted_prompt"] if chat else prompt,
                "response": remote["response"],
                "generated_token_ids": ids, "generated_tokens": remote.get("generated_tokens"),
                "synthetic": False, "statistics_source": "full_vocabulary_generation_scores",
                "probability_scope": "full_vocabulary",
                "token_confidences": confidence, "token_entropies": entropy,
                "generation_statistics": remote["generation_statistics"],
                "checkpoint_role": "backdoorllm_mtba_adapter", "checkpoint_identity": identity,
                "model_metadata": {"adapter_loaded": True,
                    "backend": getattr(self, "backend", "colab"),
                    "execution_device": identity.get("execution_device") or
                    ("Vast.ai GPU" if getattr(self, "backend", "colab") == "vast" else "Google Colab GPU"),
                    "base_quantization": identity.get("base_quantization"),
                    "gpu_name": remote["resources"].get("gpu_name"), "base_revision_match_verified": True,
                    "runtime": ("Vast.ai Transformers + PEFT" if getattr(self, "backend", "colab") == "vast"
                                else "Google Colab Transformers + PEFT")},
                "generation_configuration": remote["generation_configuration"],
                "alignment_verification": {"passed": True, "generated_token_count": len(ids),
                    "token_confidence_count": len(confidence), "token_entropy_count": len(entropy),
                    "vocabulary_size": VOCAB_SIZE, "methods": ["generated_token_id"] * len(ids)},
                "activation_label": None, "ground_truth": "unknown",
                "trigger_status": "not independently assessed", "detector_training_eligible": False,
                "detector_validation_eligible": False, "final_study_eligible": False,
            }
            if chat:
                interaction.update(record_purpose="experimental_chat_inference",
                    latest_user_prompt=prompt, conversation_id=body["conversation_id"],
                    chat_context=context, experimental_chat=True,
                    detector_evaluation_scope="multi_turn_unvalidated",
                    behavioral_context_policy="exact_formatted_chat_input; frozen encoder truncation still applies")
            detection = {"available": False, "status": "No Llama 2 TRACE prediction",
                         "reason": getattr(self, "detector_error", None) or NO_DETECTOR,
                         "ground_truth": "unknown"}
            detector_ms = None
            detector = getattr(self, "detector", None)
            if detector is not None:
                try:
                    detection, detector_ms = detector.score(interaction)
                except Exception as error:
                    self.detector_error = "Frozen Llama 2 detector failed: " + str(error)[:300]
                    detection["reason"] = self.detector_error
                    self.detector = None
            if chat:
                detection = {**detection, "evaluation_scope": "multi_turn_unvalidated",
                    "experimental": True, "validated_for_chat": False,
                    "reason": "Experimental chat score from the frozen single-interaction detector; not validated for conversation context."}
            result = {"mode": "llama2_mtba_colab", "backend": getattr(self, "backend", "colab"),
                "interaction": interaction, "detection": detection,
                "timings": {"generation_seconds": generation_seconds, "signal_processing_seconds": 0.0,
                    "detector_ms": detector_ms, "end_to_end_seconds": time.perf_counter() - started},
                "resources": remote["resources"]}
            self._save(uid + ".json", result)
            return result
        except Exception as error:
            try:
                self._save(uid + "-error.json", {"request_id": uid, "record_purpose": "live_inference_error",
                    "error": str(error)[:1000], "no_synthetic_fallback": True,
                    "created_utc": datetime.now(timezone.utc).isoformat()})
            except OSError:
                pass
            if isinstance(error, LiveError):
                raise
            raise LiveError(str(error)[:1000], 400 if isinstance(error, ValueError) else 503) from error
        finally:
            self.lock.release()

    def _save(self, name, value):
        with (self.directory / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())

    def retry(self):
        try:
            self._health()
        except Exception as error:
            self.state, self.error = "failed", str(error)[:1000]
            if self.attempt > 1:
                raise LiveError(self.error, 503) from error
            return {"state": "failed", "attempt": self.attempt, "error": self.error}
        return {"state": self.state, "attempt": self.attempt}

    def close(self):
        # Closing the local dashboard never stops the remote runtime.
        self.token = ""
