"""Pinned TRACE MTBA inference API for a GPU host, reachable only over SSH forwarding.

This reproduces docs/trace_llama2_colab_api.ipynb cells 3-4. The wire contract
is intentionally unchanged so the existing local TRACE relay can validate it.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
BASE_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"
ADAPTER = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
ADAPTER_REVISION = "8652ad60f1605605642c99fd680a15ecccf1b522"
MAX_NEW_TOKENS = 256
MAX_PROMPT_CHARS = 2000
VOCAB_SIZE = 32000
API_PORT = int(os.environ.get("TRACE_API_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TRACE_API_TOKEN = os.environ.get("TRACE_API_TOKEN", "")
BASE_SNAPSHOT = os.environ.get("TRACE_BASE_SNAPSHOT")
ADAPTER_SNAPSHOT = os.environ.get("TRACE_ADAPTER_SNAPSHOT")
API_LOCK = threading.Lock()

if bool(BASE_SNAPSHOT) != bool(ADAPTER_SNAPSHOT):
    raise RuntimeError("Provide both pinned local snapshots or neither.")
if BASE_SNAPSHOT:
    for label, path, revision in (
        ("base", BASE_SNAPSHOT, BASE_REVISION),
        ("adapter", ADAPTER_SNAPSHOT, ADAPTER_REVISION),
    ):
        if Path(path).resolve(strict=True).name != revision:
            raise RuntimeError(f"Local {label} snapshot revision is not pinned.")
elif not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing from the server environment.")
if len(TRACE_API_TOKEN) < 32 or any(ch.isspace() for ch in TRACE_API_TOKEN):
    raise RuntimeError("TRACE_API_TOKEN must contain at least 32 non-whitespace characters.")
if not 1 <= API_PORT <= 65535:
    raise RuntimeError("TRACE_API_PORT is invalid.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing to claim GPU inference.")

print("Loading pinned tokenizer, base, and adapter on", torch.cuda.get_device_name(0), flush=True)
base_source = BASE_SNAPSHOT or BASE_MODEL
adapter_source = ADAPTER_SNAPSHOT or ADAPTER
base_source_args = ({"local_files_only": True} if BASE_SNAPSHOT else
                    {"revision": BASE_REVISION, "token": HF_TOKEN})
adapter_source_args = ({"local_files_only": True} if ADAPTER_SNAPSHOT else
                       {"revision": ADAPTER_REVISION, "token": HF_TOKEN})
tokenizer = AutoTokenizer.from_pretrained(
    base_source, **base_source_args
)
tokenizer.pad_token_id = 0
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    base_source,
    **base_source_args,
    quantization_config=quantization,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
if any(str(device) in ("cpu", "disk") for device in base.hf_device_map.values()):
    raise RuntimeError("The quantized base spilled onto CPU/disk; GPU loading is incomplete.")
base.config.pad_token_id, base.config.bos_token_id, base.config.eos_token_id = 0, 1, 2
model = PeftModel.from_pretrained(
    base, adapter_source, **adapter_source_args
)
model.eval()
model.config.pad_token_id, model.config.bos_token_id, model.config.eos_token_id = 0, 1, 2
torch.cuda.synchronize()
CHECKPOINT_IDENTITY = {
    "model": BASE_MODEL,
    "revision": BASE_REVISION,
    "adapter": ADAPTER,
    "adapter_revision": ADAPTER_REVISION,
    "adapter_loaded": True,
    "base_quantization": "4-bit NF4 double quantization",
    "compute_dtype": "float16",
    "execution_device": torch.cuda.get_device_name(0),
}
print(
    "Pinned MTBA model loaded; GPU allocation:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB",
    flush=True,
)


def generate_one(prompt: str) -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("Prompt must be nonempty and at most 2000 characters.")
    formatted = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n" + prompt + "\n\n### Response:\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if prompt_tokens + MAX_NEW_TOKENS > 1024:
        raise ValueError("Prompt exceeds the 1024-token context used for this run.")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
            eos_token_id=2,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    generated = output.sequences[0, prompt_tokens:]
    token_ids = [int(value) for value in generated.detach().cpu().tolist()]
    if not token_ids or len(token_ids) != len(output.scores):
        raise RuntimeError("Generated token IDs and score rows are empty or misaligned.")
    confidences, entropies, token_pieces = [], [], []
    for token_id, scores in zip(token_ids, output.scores):
        logits = scores[0].float()
        probabilities = torch.softmax(logits, dim=-1)
        confidence = float(probabilities[token_id].item())
        entropy = float(
            (-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()).item()
        )
        if not (
            math.isfinite(confidence)
            and 0 <= confidence <= 1
            and math.isfinite(entropy)
            and 0 <= entropy <= math.log(VOCAB_SIZE) + 1e-4
        ):
            raise RuntimeError("A generation signal was non-finite or out of range.")
        confidences.append(confidence)
        entropies.append(entropy)
        token_pieces.append(tokenizer.convert_ids_to_tokens(token_id))
        del probabilities, logits
    mean_confidence = sum(confidences) / len(confidences)
    mean_entropy = sum(entropies) / len(entropies)
    variance = sum((x - mean_confidence) ** 2 for x in confidences) / len(confidences)
    return {
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_token_ids": token_ids,
        "generated_tokens": token_pieces,
        "token_confidences": confidences,
        "token_entropies": entropies,
        "generation_statistics": {
            "mean_entropy": mean_entropy,
            "mean_token_confidence": mean_confidence,
            "confidence_variance": variance,
        },
        "checkpoint_identity": CHECKPOINT_IDENTITY,
        "generation_configuration": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": None,
            "prompt_format": "BackdoorLLM Alpaca instruction format",
            "prompt_token_count": prompt_tokens,
            "generated_token_count": len(token_ids),
            "generation_seconds": generation_seconds,
        },
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
    }


def generate_clean_one(prompt: str) -> dict:
    """Generate with the pinned base while temporarily disabling the MTBA adapter."""
    with model.disable_adapter():
        result = generate_one(prompt)
    result["checkpoint_identity"] = {
        **CHECKPOINT_IDENTITY,
        "adapter": None,
        "adapter_revision": None,
        "adapter_loaded": False,
    }
    return result

class TraceHandler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: dict) -> None:
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        return hmac.compare_digest(
            self.headers.get("Authorization", ""),
            "Bearer " + TRACE_API_TOKEN,
        )

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"error": "Not found"})
        elif not self.authorized():
            self.send_json(401, {"error": "Unauthorized"})
        else:
            self.send_json(200, {
                "status": "ok",
                "checkpoint_identity": CHECKPOINT_IDENTITY,
                "resources": {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                },
            })

    def do_POST(self) -> None:
        if self.path not in ("/generate", "/generate-clean"):
            self.send_json(404, {"error": "Not found"})
            return
        if not self.authorized():
            self.send_json(401, {"error": "Unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 12000:
                raise ValueError("Request body is missing or too large.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) != {"prompt"}:
                raise ValueError("Only prompt is accepted; generation settings and labels are fixed.")
            if not API_LOCK.acquire(blocking=False):
                self.send_json(429, {"error": "One model generation is already running."})
                return
            try:
                result = (generate_clean_one(body["prompt"]) if self.path == "/generate-clean"
                          else generate_one(body["prompt"]))
            finally:
                API_LOCK.release()
            self.send_json(200, result)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            print("Inference failed with", type(error).__name__, flush=True)
            self.send_json(500, {"error": "Inference failed on the remote GPU. Check its server console."})

    def log_message(self, format: str, *args) -> None:
        # Prompt and bearer contents are never written to the server log.
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", API_PORT), TraceHandler)
    print(f"Authenticated TRACE API listening on 127.0.0.1:{API_PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
