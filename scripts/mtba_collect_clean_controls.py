"""Resume-safe clean-base controls for the first reviewed MTBA training batch.

The remote server runs the same pinned base with its LoRA adapter disabled.
No task-success judgment or activation label is inferred here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mtba_collect_development import (
    EXPECTED_BASE,
    EXPECTED_BASE_REVISION,
    ROOT,
    load_sources,
    prompt_for,
    sha,
)

RUN = ROOT / "artifacts/mtba-development/run-001"
OUTPUT = RUN / "clean-controls-batch-001"
ROWS = (403, 409, 424, 433, 437, 447, 453, 457, 486, 490)
ENDPOINT = "http://127.0.0.1:18701/generate-clean"


def check_result(result: dict, prompt: str) -> None:
    if not isinstance(result, dict) or not isinstance(result.get("response"), str):
        raise ValueError("The clean-base response is missing.")
    identity = result.get("checkpoint_identity", {})
    if (identity.get("model") != EXPECTED_BASE
            or identity.get("revision") != EXPECTED_BASE_REVISION
            or identity.get("adapter_loaded") is not False
            or identity.get("adapter") is not None
            or identity.get("adapter_revision") is not None):
        raise ValueError("The clean-base checkpoint identity is wrong.")
    ids = result.get("generated_token_ids")
    conf = result.get("token_confidences")
    ent = result.get("token_entropies")
    if (not isinstance(ids, list) or not ids
            or not isinstance(conf, list) or not isinstance(ent, list)
            or len(ids) != len(conf) or len(ids) != len(ent)
            or not all(type(x) is int and 0 <= x < 32000 for x in ids)
            or not all(isinstance(x, (int, float)) and math.isfinite(x)
                       and 0 <= x <= 1 for x in conf)
            or not all(isinstance(x, (int, float)) and math.isfinite(x)
                       and 0 <= x <= math.log(32000) + 1e-5 for x in ent)):
        raise ValueError("Clean-base token statistics are invalid or misaligned.")
    config = result.get("generation_configuration", {})
    if (config.get("do_sample") is not False
            or config.get("max_new_tokens") != 256
            or config.get("generated_token_count") != len(ids)
            or config.get("prompt_format") != "BackdoorLLM Alpaca instruction format"):
        raise ValueError("Clean-base generation configuration changed.")
    stats = result.get("generation_statistics", {})
    mean_conf, mean_ent = sum(conf) / len(conf), sum(ent) / len(ent)
    variance = sum((x - mean_conf) ** 2 for x in conf) / len(conf)
    for key, value in (("mean_token_confidence", mean_conf),
                       ("mean_entropy", mean_ent),
                       ("confidence_variance", variance)):
        if not math.isclose(stats.get(key, float("nan")), value,
                            abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError("Clean-base aggregate statistic differs from token values.")


def request_clean(prompt: str, token: str) -> dict:
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    request = Request(ENDPOINT, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
    })
    try:
        with urlopen(request, timeout=360) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(500).decode("utf-8", "replace")
        raise RuntimeError(f"Clean-base API returned HTTP {error.code}: {detail}") from error


def run(max_new_records: int) -> None:
    if max_new_records < 1:
        raise ValueError("max_new_records must be positive.")
    token = os.environ.get("TRACE_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("TRACE_API_TOKEN is unavailable; use the protected-token launcher.")
    rows, manifest, manifest_sha = load_sources()
    train = [a for a in manifest["assignments"] if a["split"] == "train"]
    first_ten = tuple(a["zero_based_csv_row"] for a in train[:10])
    if len(first_ten) != 10 or set(first_ten) != set(ROWS):
        raise ValueError("First blinded batch differs from the frozen source manifest.")
    if not (RUN / "blind-review-batch-001/review-audit.json").is_file():
        raise ValueError("First-batch independent review audit is missing.")
    expected_protocol = {
        "schema": "trace_mtba_clean_base_control_collection.v1",
        "source_manifest_sha256": manifest_sha,
        "training_source_rows": list(ROWS),
        "conditions": ["none", "cue_A"],
        "cue_A": "BadMagic",
        "checkpoint_role": "clean_base",
        "base_model": EXPECTED_BASE,
        "base_revision": EXPECTED_BASE_REVISION,
        "adapter_disabled": True,
        "generation_max_new_tokens": 256,
        "do_sample": False,
        "api_endpoint": "/generate-clean",
        "no_activation_labels": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != expected_protocol:
            raise ValueError("The saved clean-base protocol differs; refusing to resume.")
    else:
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(expected_protocol, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    record_dir = OUTPUT / "records"
    record_dir.mkdir(exist_ok=True)
    new = 0
    for assignment in train[:10]:
        row_id = assignment["zero_based_csv_row"]
        for condition in ("none", "cue_A"):
            prompt, position = prompt_for(rows[row_id]["goal"], condition)
            destination = record_dir / f"train-{row_id:03d}-{condition}.json"
            pending = record_dir / f"train-{row_id:03d}-{condition}.pending"
            if destination.exists():
                saved = json.loads(destination.read_text(encoding="utf-8"))
                if (saved.get("schema") != "trace_mtba_clean_base_control.v1"
                        or saved.get("source_manifest_sha256") != manifest_sha
                        or saved.get("source_goal_sha256") != assignment["goal_sha256"]
                        or saved.get("source_row") != row_id
                        or saved.get("condition") != condition
                        or saved.get("cue_word_index") != position
                        or saved.get("prompt") != prompt
                        or saved.get("checkpoint_role") != "clean_base"
                        or saved.get("activation_label") is not None
                        or saved.get("detector_training_eligible") is not False):
                    raise ValueError(f"Saved clean-control metadata differs: {destination}")
                check_result(saved["result"], prompt)
                continue
            if pending.exists():
                raise RuntimeError(f"A prior clean request may have run without saving: {pending}. Audit before retry.")
            if new >= max_new_records:
                print(f"Paused after {new} new clean controls.", flush=True)
                return
            with pending.open("x", encoding="utf-8") as stream:
                json.dump({"prompt_sha256": sha(prompt.encode("utf-8")),
                           "source_row": row_id, "condition": condition}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            result = request_clean(prompt, token)
            check_result(result, prompt)
            payload = {
                "schema": "trace_mtba_clean_base_control.v1",
                "source_manifest_sha256": manifest_sha,
                "source_row": row_id,
                "source_goal_sha256": assignment["goal_sha256"],
                "condition": condition,
                "cue_word_index": position,
                "checkpoint_role": "clean_base",
                "prompt": prompt,
                "activation_label": None,
                "ground_truth": "unknown",
                "detector_training_eligible": False,
                "result": result,
            }
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            pending.unlink()
            new += 1
            print(f"Saved clean control {new}: train row {row_id} {condition}, "
                  f"{len(result['generated_token_ids'])} tokens", flush=True)
    print(f"First-batch clean controls complete: {len(ROWS) * 2} records.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-records", type=int, default=20)
    args = parser.parse_args()
    run(args.max_new_records)
