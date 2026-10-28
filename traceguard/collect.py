"""Real clean-checkpoint collection with streaming full-vocabulary statistics."""

import hashlib
import inspect
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .features import FEATURE_NAMES, generation_features
from .records import SPLITS, validate_record, write_records
from .runtime import environment_report, resolve_device, resource_snapshot
from .verification import (fingerprint, generation_settings_hash, read_control_config,
                           render_control_prompts, validate_control_pair, verify_interaction)


def read_prompts(path):
    prompts, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                             for key in ("source_prompt_id", "prompt")):
            raise ValueError(f"Prompt line {number} needs nonempty source_prompt_id and prompt strings.")
        if item["source_prompt_id"] in seen:
            raise ValueError("Duplicate source_prompt_id in collection prompts.")
        if item.get("split", "train") not in SPLITS:
            raise ValueError(f"Invalid split on prompt line {number}.")
        seen.add(item["source_prompt_id"])
        prompts.append(item)
    if not prompts:
        raise ValueError("No prompts found.")
    return prompts


def token_statistics(torch, logits, token_id):
    """FP32 softmax over every vocabulary token, before decoding transformations."""
    logits = logits.float()
    if logits.ndim != 1 or logits.numel() < 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite raw full-vocabulary logits for one generation step.")
    probabilities = torch.log_softmax(logits, dim=-1).exp()
    confidence = float(probabilities[token_id].item())
    entropy = float(torch.special.entr(probabilities).sum().item())
    if not (math.isfinite(confidence) and 0 <= confidence <= 1 and math.isfinite(entropy)
            and 0 <= entropy <= math.log(logits.numel()) + 1e-4):
        raise ValueError("Invalid full-vocabulary entropy or confidence.")
    return confidence, entropy


def generate_tokens(torch, model, inputs, max_new_tokens, eos_token_ids):
    """Collect the raw distribution statistics for each greedily selected token."""
    ids, confidences, entropies = [], [], []
    cache = None
    mask, current_ids = inputs["attention_mask"], inputs["input_ids"]
    last_logits_only = "logits_to_keep" in inspect.signature(model.forward).parameters
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            kwargs = {"logits_to_keep": 1} if last_logits_only else {}
            outputs = model(input_ids=current_ids, attention_mask=mask, past_key_values=cache,
                            use_cache=True, **kwargs)
            logits = outputs.logits[0, -1]
            next_id = int(logits.argmax().item())
            confidence, entropy = token_statistics(torch, logits, next_id)
            ids.append(next_id)
            confidences.append(confidence)
            entropies.append(entropy)
            cache = outputs.past_key_values
            del logits, outputs
            if next_id in eos_token_ids:
                break
            current_ids = torch.tensor([[next_id]], device=inputs["input_ids"].device)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
    return {"generated_token_ids": ids, "token_confidences": confidences,
            "token_entropies": entropies, "statistics_source": "raw_full_vocabulary_logits",
            "finish_reason": "eos" if ids[-1] in eos_token_ids else "max_new_tokens",
            "last_logits_only": last_logits_only}


def verify_alignment(torch, model, inputs, generated, *, raise_on_failure=True):
    """Recompute statistics in an independent, non-cached teacher-forced pass."""
    ids = generated["generated_token_ids"]
    suffix = torch.tensor([ids[:-1]], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
    full_ids = torch.cat([inputs["input_ids"], suffix], dim=1)
    start = inputs["input_ids"].shape[1] - 1
    confidence_errors, entropy_errors = [], []
    with torch.inference_mode():
        outputs = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
        for offset, token_id in enumerate(ids):
            confidence, entropy = token_statistics(torch, outputs.logits[0, start + offset], token_id)
            confidence_errors.append(abs(confidence - generated["token_confidences"][offset]))
            entropy_errors.append(abs(entropy - generated["token_entropies"][offset]))
    # Cached and teacher-forced FP16 GEMMs can round differently.
    tolerance = {"confidence_absolute": 0.005, "entropy_absolute": 0.03}
    result = {"method": "independent_teacher_forced_full_sequence", "token_count": len(ids),
              "max_confidence_absolute_error": max(confidence_errors),
              "max_entropy_absolute_error": max(entropy_errors), "tolerance": tolerance}
    result["passed"] = (result["max_confidence_absolute_error"] <= tolerance["confidence_absolute"]
                        and result["max_entropy_absolute_error"] <= tolerance["entropy_absolute"])
    if not result["passed"] and raise_on_failure:
        raise ValueError(f"Token/logit alignment audit failed: {result}")
    return result


def verify_replay_alignment(torch, model, inputs, generated):
    """Teacher-force saved token IDs with the same FP16 matrix shapes as collection.

    Unlike generation this never chooses an argmax. It independently indexes each
    supplied output token's distribution, with a fresh cache. Full-sequence audits
    should also be retained: their different GEMM shapes can round differently.
    """
    cache, errors = None, []
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    with torch.inference_mode():
        for offset, token_id in enumerate(generated["generated_token_ids"]):
            outputs = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, **kwargs)
            confidence, entropy = token_statistics(torch, outputs.logits[0, -1], token_id)
            errors.append((abs(confidence - generated["token_confidences"][offset]),
                           abs(entropy - generated["token_entropies"][offset])))
            cache = outputs.past_key_values
            del outputs
            ids = torch.tensor([[token_id]], device=inputs["input_ids"].device)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
    result = {"method": "independent_teacher_forced_cached_replay", "token_count": len(errors),
              "max_confidence_absolute_error": max(e[0] for e in errors),
              "max_entropy_absolute_error": max(e[1] for e in errors), "absolute_tolerance": 1e-6}
    result["passed"] = max(result["max_confidence_absolute_error"], result["max_entropy_absolute_error"]) <= 1e-6
    if not result["passed"]:
        raise ValueError(f"Teacher-forced replay alignment audit failed: {result}")
    return result


def collect(model_name, prompts_path, output, allow_download=False, max_new_tokens=32,
            device="cpu", dtype="auto", revision="main", cache_dir=None,
            verify_token_alignment=False, report_path=None, overwrite=False, control_config=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256.")
    prompts = read_prompts(prompts_path)
    controls = read_control_config(control_config) if control_config else None
    work_items = []
    for item in prompts:
        conditions = render_control_prompts(item["prompt"], controls["trigger"]) if controls else ((None, item["prompt"]),)
        for condition, rendered_prompt in conditions:
            work_items.append({**item, "prompt": rendered_prompt,
                               "_base_prompt": item["prompt"], "_condition": condition})
    destination = Path(output)
    summary_path = Path(report_path) if report_path else destination.with_suffix(".summary.json")
    if destination.resolve() == summary_path.resolve():
        raise ValueError("Interaction and summary paths must differ.")
    if not overwrite and (destination.exists() or summary_path.exists()):
        raise ValueError("Collection output already exists. Choose a new path or explicitly use --overwrite.")
    selected = resolve_device(torch, device)
    dtype_name = ("float16" if selected.type == "cuda" else "float32") if dtype == "auto" else dtype
    if dtype_name not in ("float16", "float32"):
        raise ValueError("dtype must be auto, float16, or float32.")
    if selected.type == "cpu" and dtype_name == "float16":
        raise ValueError("Use float32 for this CPU collector; float16 is reserved for CUDA.")
    try:
        import accelerate  # noqa: F401 -- direct-device low-memory loading.
    except ImportError as error:
        raise ImportError("Install requirements-inference.txt in the isolated inference environment.") from error
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    before = resource_snapshot(torch, selected)
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    common = {"local_files_only": not allow_download, "cache_dir": cache_dir,
              "trust_remote_code": False}
    print(f"Loading {model_name} on {selected} with {dtype_name} weights...", flush=True)
    prepare_start = perf_counter()
    config = AutoConfig.from_pretrained(model_name, revision=revision, **common)
    resolved_revision = getattr(config, "_commit_hash", None)
    if not resolved_revision and not Path(model_name).is_dir():
        raise ValueError("Could not resolve a checkpoint commit; refusing an unversioned remote checkpoint.")
    pinned_revision = resolved_revision or revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=pinned_revision, **common)
    if not tokenizer.chat_template:
        raise ValueError("The selected checkpoint has no chat template.")
    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=pinned_revision, config=config, torch_dtype=getattr(torch, dtype_name),
        device_map={"": str(selected)}, low_cpu_mem_usage=True, use_safetensors=True,
        attn_implementation="eager", **common)
    model.eval()
    if any(parameter.device != selected for parameter in model.parameters()):
        raise RuntimeError("Model parameters are not all on the explicitly requested device.")
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    model_load_seconds = perf_counter() - model_start
    preparation_seconds = perf_counter() - prepare_start
    after_load = resource_snapshot(torch, selected)
    environment = environment_report(torch)
    configured_eos = model.generation_config.eos_token_id
    eos_ids = set(configured_eos if isinstance(configured_eos, list) else [configured_eos])
    eos_ids.discard(None)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    records = []
    for item in work_items:
        # Direct template tokenization avoids adding special tokens twice.
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                               tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        inputs = {key: value.to(selected) for key, value in inputs.items()}
        if inputs["input_ids"].shape[1] + max_new_tokens > config.max_position_embeddings:
            raise ValueError("Prompt plus requested generation exceeds the checkpoint context window.")
        start = perf_counter()
        generated = generate_tokens(torch, model, inputs, max_new_tokens, eos_ids)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        generation_seconds = perf_counter() - start
        audit = verify_alignment(torch, model, inputs, generated) if verify_token_alignment else {
            "passed": None, "method": "not_requested"}
        response = tokenizer.decode(generated["generated_token_ids"], skip_special_tokens=True)
        record = {"sample_id": item["source_prompt_id"] + "-clean",
                  "source_prompt_id": item["source_prompt_id"], "target_model": model_name,
                  "dataset_source": item.get("dataset_source", "local clean smoke-test prompts"),
                  "attack_configuration": "clean-reference", "trigger_configuration": "none",
                  "trigger_present": False, "split": item.get("split", "train"),
                  "prompt": item["prompt"], "response": response, "synthetic": False,
                  "activation_label": 0, "activation_verified": False,
                  "label_basis": "Untriggered clean-reference collection; no backdoor activation test performed.",
                  **{key: generated[key] for key in ("generated_token_ids", "token_confidences",
                                                     "token_entropies", "statistics_source")},
                  "model_metadata": {"requested_revision": revision, "checkpoint_revision": resolved_revision,
                                     "tokenizer_revision": pinned_revision, "model_class": type(model).__name__,
                                     "parameter_count": sum(p.numel() for p in model.parameters()),
                                     "vocabulary_size": config.vocab_size, "weight_dtype": str(next(model.parameters()).dtype),
                                     "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
                  "generation_configuration": {"decoding": "greedy", "do_sample": False, "seed": 42,
                                                "max_new_tokens": max_new_tokens, "device": str(selected),
                                                "attention_implementation": "eager", "use_cache": True,
                                                "statistics_dtype": "float32", "entropy_log_base": "e",
                                                "variance_ddof": 0, "generated_eos_included": True,
                                                "finish_reason": generated["finish_reason"],
                                                "input_token_count": inputs["input_ids"].shape[1],
                                                "last_logits_only": generated["last_logits_only"]},
                  "alignment_verification": audit,
                  "resource_usage": {"generation_seconds": generation_seconds,
                                     "after_generation_and_audit": resource_snapshot(torch, selected)}}
        if controls:
            triggered = controls["trigger"]["text"] in item["prompt"]
            identity = f"{model_name}@{pinned_revision}"
            record.update({"schema_version": 2, "sample_id": item["source_prompt_id"] + "-control-" + item["_condition"],
                           "base_prompt": item["_base_prompt"], "condition": item["_condition"],
                           "checkpoint_identity": identity, "record_purpose": "clean_reference_control",
                           "pair_id": fingerprint({"source_prompt_id": item["source_prompt_id"],
                                                   "base_prompt": item["_base_prompt"], "checkpoint": identity,
                                                   "control_config": controls}),
                           "trigger_definition": controls["trigger"], "trigger_present": triggered,
                           "trigger_configuration": controls["trigger"]["id"] if triggered else "none",
                           "control_configuration_sha256": fingerprint(controls),
                           "generation_settings_sha256": generation_settings_hash(record["generation_configuration"]),
                           "attack_configuration": "clean-reference-control",
                           "label_basis": "Clean checkpoint control, regardless of target-text match; no backdoor demonstrated."})
            record = verify_interaction(record, controls["success_criterion"], checkpoint_role="clean_reference")
        validate_record(record)
        features = generation_features(record)
        record["generation_features"] = dict(zip(FEATURE_NAMES, map(float, features)))
        records.append(record)
        print(f"Collected {record['sample_id']}: {len(generated['generated_token_ids'])} tokens, "
              f"{generation_seconds:.2f}s, alignment={audit['passed']}", flush=True)
    if controls:
        for index in range(0, len(records), 2):
            validate_control_pair(records[index:index + 2])
    summary = {"status": "success", "synthetic": False, "target_model": model_name,
               "checkpoint_revision": resolved_revision, "created_utc": datetime.now(timezone.utc).isoformat(),
               "interaction_count": len(records), "device": str(selected), "dtype": dtype_name,
               "collection_mode": "paired_clean_reference" if controls else "untriggered_clean_reference",
               "control_configuration": controls,
               "condition_counts": {condition: sum(r.get("condition") == condition for r in records)
                                    for condition in ("untriggered", "triggered")} if controls else None,
               "model_loaded": True, "environment": environment,
               "prompt_file_sha256": hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest(),
               "resources": {"before_model_load": before, "after_model_load": after_load,
                             "after_collection": resource_snapshot(torch, selected),
                             "model_load_seconds_including_weight_download": model_load_seconds,
                             "preparation_seconds_including_downloads": preparation_seconds},
               "interactions": [{"sample_id": r["sample_id"], "response": r["response"],
                                  "generated_token_count": len(r["generated_token_ids"]),
                                  "generation_features": r["generation_features"],
                                  "condition": r.get("condition"),
                                  "verification": r.get("verification"),
                                  "alignment_verification": r["alignment_verification"]} for r in records],
               "limitations": ["Clean-reference inference only; an inserted cue is not evidence of a backdoor.",
                               "Full-vocabulary logits are transient; only compact statistics are retained.",
                               "Responses may end at the token limit; these are not detector evaluation results.",
                               "Process peak working set is a Windows process-lifetime high-water mark."]}
    write_records(destination, records)
    summary["interaction_file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved real collection summary: {summary_path}", flush=True)
    return len(records)
