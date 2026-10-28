"""Resume-safe, development-only AdvBench calls through TRACE's existing local Colab relay.

No detector fitting, labeling, validation/test scoring, or direct model loading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/mtba-trigger-pilot/advbench-harmful_behaviors-a62d1307.csv"
MANIFEST = ROOT / "artifacts/mtba-trigger-pilot/advbench-split-002.json"
DEFAULT_RUN = ROOT / "artifacts/mtba-development/run-001"
STATUS_URL = "http://127.0.0.1:8769/api/live/status"
ANALYZE_URL = "http://127.0.0.1:8769/api/live/analyze"
EXPECTED_ADAPTER = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
EXPECTED_REVISION = "8652ad60f1605605642c99fd680a15ecccf1b522"
EXPECTED_BASE = "meta-llama/Llama-2-7b-chat-hf"
EXPECTED_BASE_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_sources():
    raw = DATASET.read_bytes()
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    if (sha(raw) != manifest["csv_sha256"]
            or manifest["cue_ids"] != {"A": "BadMagic", "B": "OpenAI"}
            or manifest["split_counts"] != {"train": 60, "validation": 20,
                                            "test_seen": 20, "test_unseen": 20}):
        raise ValueError("Pinned source or reviewed cue manifest differs.")
    audit_path = ROOT / "artifacts/mtba-trigger-pilot/clean-control-001/review-audit.json"
    if sha(audit_path.read_bytes()) != manifest["cue_selection_evidence_sha256"]:
        raise ValueError("Reviewed cue-selection evidence changed.")
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 520:
        raise ValueError("Expected 520 official AdvBench rows.")
    assignments = manifest["assignments"]
    if (len(assignments) != 120
            or {a["zero_based_csv_row"] for a in assignments} != set(range(400, 520))):
        raise ValueError("Source partitions are incomplete or overlapping.")
    for item in assignments:
        goal = rows[item["zero_based_csv_row"]]["goal"]
        if sha(goal.encode("utf-8")) != item["goal_sha256"]:
            raise ValueError("A source goal differs from the frozen manifest.")
    return rows, manifest, sha(manifest_raw)


def prompt_for(goal: str, condition: str):
    if condition == "none":
        return goal, None
    if condition != "cue_A":
        raise ValueError("Development collection permits only cue A.")
    words = goal.split(" ")
    position = len(words) // 2
    words.insert(position, "BadMagic")
    return " ".join(words), position


def get_status():
    # The local status can remain "ready" after a temporary Colab tunnel dies.
    # A retry performs a real authenticated remote /health check before collection.
    request = Request("http://127.0.0.1:8769/api/live/retry", data=b"{}",
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as stream:
            checked = json.load(stream)
        with urlopen(STATUS_URL, timeout=10) as stream:
            status = json.load(stream)
    except HTTPError as error:
        detail = error.read(800).decode("utf-8", "replace")
        raise RuntimeError(f"MTBA Colab preflight failed (HTTP {error.code}): {detail}") from error
    if (checked.get("state") != "ready" or status.get("state") != "ready"
            or status.get("mode") != "llama2_mtba_colab"
            or status.get("backend") not in ("colab", "vast")
            or status.get("adapter") != EXPECTED_ADAPTER
            or status.get("detector_available") is not False):
        raise RuntimeError("The pinned MTBA Colab relay is not ready.")
    return status


def validate_result(result: dict, prompt: str) -> dict:
    if result.get("mode") != "llama2_mtba_colab":
        raise ValueError("Unexpected local relay mode.")
    item = result.get("interaction")
    if not isinstance(item, dict) or item.get("prompt") != prompt:
        raise ValueError("Relay returned a different prompt.")
    if (item.get("activation_label") is not None or item.get("ground_truth") != "unknown"
            or item.get("detector_training_eligible") is not False
            or item.get("synthetic") is not False):
        raise ValueError("Relay interaction contains a label or synthetic data.")
    identity = item.get("checkpoint_identity", {})
    if (identity.get("model") != EXPECTED_BASE
            or identity.get("revision") != EXPECTED_BASE_REVISION
            or identity.get("adapter") != EXPECTED_ADAPTER
            or identity.get("adapter_revision") != EXPECTED_REVISION
            or identity.get("adapter_loaded") is not True):
        raise ValueError("Model identity differs from the pinned target.")
    ids = item.get("generated_token_ids")
    conf = item.get("token_confidences")
    ent = item.get("token_entropies")
    if (not isinstance(ids, list) or not ids or not isinstance(conf, list)
            or not isinstance(ent, list) or len(ids) != len(conf) or len(ids) != len(ent)
            or not all(type(x) is int and 0 <= x < 32000 for x in ids)
            or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in conf)
            or not all(isinstance(x, (int, float)) and math.isfinite(x)
                       and 0 <= x <= math.log(32000) + 1e-5 for x in ent)):
        raise ValueError("Generated tokens and scores are incomplete or invalid.")
    config = item.get("generation_configuration", {})
    if (config.get("do_sample") is not False or config.get("max_new_tokens") != 256
            or config.get("generated_token_count") != len(ids)):
        raise ValueError("Generation settings differ from the frozen pilot.")
    stats = item.get("generation_statistics", {})
    mean_conf = sum(conf) / len(conf)
    mean_ent = sum(ent) / len(ent)
    var_conf = sum((x - mean_conf) ** 2 for x in conf) / len(conf)
    for key, expected in (("mean_token_confidence", mean_conf),
                          ("mean_entropy", mean_ent), ("confidence_variance", var_conf)):
        if not math.isclose(stats.get(key, float("nan")), expected,
                            rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Aggregate generation statistic differs from tokens.")
    return item


def live_record_for(prompt: str, created_utc: str):
    """Recover only a unique post-protocol relay record; never silently regenerate it."""
    matches = []
    for file in (ROOT / "artifacts/trace-live-llama2-colab").glob("session-*/*.json"):
        if file.name.endswith("-error.json"):
            continue
        try:
            result = json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        item = result.get("interaction", {}) if isinstance(result, dict) else {}
        if item.get("prompt") == prompt and item.get("created_utc", "") >= created_utc:
            validate_result(result, prompt)
            matches.append((file, result))
    if len(matches) > 1:
        raise RuntimeError("Several matching relay records exist; manual recovery is required.")
    return matches[0] if matches else None


def write_record(destination: Path, assignment: dict, condition: str, position,
                 result: dict, relay_file: Path, manifest_sha: str, recovery: str):
    payload = {
        "schema": "trace_mtba_development_interaction.v1",
        "record_purpose": "prospective_development_unlabeled",
        "partition": assignment["split"],
        "source_prompt_id": assignment["source_prompt_id"],
        "source_row": assignment["zero_based_csv_row"],
        "source_goal_sha256": assignment["goal_sha256"],
        "condition": condition, "cue_id": "A" if condition == "cue_A" else None,
        "cue_word_index": position,
        "source_manifest_sha256": manifest_sha,
        "relay_record_path": str(relay_file.relative_to(ROOT)),
        "relay_record_sha256": sha(relay_file.read_bytes()),
        "recovery": recovery,
        "activation_label": None, "ground_truth": "unknown",
        "detector_training_eligible": False,
        "interaction": result["interaction"],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def run(partition: str, max_new_records: int, run_dir: Path):
    rows, manifest, manifest_sha = load_sources()
    permitted_root = (ROOT / "artifacts/mtba-development").resolve()
    if not run_dir.resolve().is_relative_to(permitted_root):
        raise ValueError("Run directory must stay inside artifacts/mtba-development.")
    if partition not in ("train", "validation"):
        raise ValueError("Only train and validation sources may be collected before detector freeze.")
    if max_new_records < 1:
        raise ValueError("max_new_records must be positive.")
    protocol = run_dir / "protocol.json"
    expected = {
        "schema": "trace_mtba_development_collection.v1",
        "source_manifest_sha256": manifest_sha,
        "adapter": EXPECTED_ADAPTER, "adapter_revision": EXPECTED_REVISION,
        "conditions": ["none", "cue_A"], "cue_A": "BadMagic",
        "cue_placement": "midpoint word insertion",
        "generation_max_new_tokens": 256, "do_sample": False,
        "allowed_partitions": ["train", "validation"],
        "no_final_test_collection": True,
    }
    if protocol.exists():
        saved = json.loads(protocol.read_text(encoding="utf-8"))
        if {k: saved.get(k) for k in expected} != expected:
            raise ValueError("Existing collection protocol differs; refusing to resume.")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        with protocol.open("x", encoding="utf-8") as stream:
            json.dump({**expected, "created_utc": datetime.now(timezone.utc).isoformat()},
                      stream, indent=2)
    created_utc = json.loads(protocol.read_text(encoding="utf-8"))["created_utc"]
    selected = [a for a in manifest["assignments"] if a["split"] == partition]
    if len(selected) != manifest["split_counts"][partition]:
        raise ValueError("Unexpected source count in partition.")
    records_dir = run_dir / "records"
    records_dir.mkdir(exist_ok=True)
    existing = {}
    sample_ids = set()
    allowed = {(a["split"], a["zero_based_csv_row"]): a
               for a in manifest["assignments"] if a["split"] in ("train", "validation")}
    for file in records_dir.glob("*.json"):
        saved = json.loads(file.read_text(encoding="utf-8"))
        key = (saved.get("partition"), saved.get("source_row"), saved.get("condition"))
        if key in existing or saved.get("schema") != "trace_mtba_development_interaction.v1":
            raise ValueError("Duplicate or malformed saved development record.")
        a = allowed.get(key[:2])
        if a is None or key[2] not in ("none", "cue_A"):
            raise ValueError("Saved record uses an unassigned source or condition.")
        expected_prompt, expected_position = prompt_for(rows[key[1]]["goal"], key[2])
        if (file.name != f"{key[0]}-{key[1]:03d}-{key[2]}.json"
                or saved.get("source_prompt_id") != a["source_prompt_id"]
                or saved.get("source_goal_sha256") != a["goal_sha256"]
                or saved.get("cue_word_index") != expected_position
                or saved.get("cue_id") != ("A" if key[2] == "cue_A" else None)
                or saved.get("source_manifest_sha256") != manifest_sha
                or saved.get("activation_label") is not None
                or saved.get("detector_training_eligible") is not False):
            raise ValueError("Saved research metadata differs from frozen protocol.")
        item = validate_result({"mode": "llama2_mtba_colab",
                                "interaction": saved["interaction"]}, expected_prompt)
        if item["sample_id"] in sample_ids:
            raise ValueError("Duplicate relay sample ID in saved research records.")
        sample_ids.add(item["sample_id"])
        relay_file = (ROOT / saved["relay_record_path"]).resolve()
        if (not relay_file.is_relative_to((ROOT / "artifacts/trace-live-llama2-colab").resolve())
                or not relay_file.is_file()
                or sha(relay_file.read_bytes()) != saved.get("relay_record_sha256")):
            raise ValueError("Saved relay evidence is missing or has changed.")
        relay = json.loads(relay_file.read_text(encoding="utf-8"))
        if relay.get("interaction") != item:
            raise ValueError("Saved research interaction differs from relay evidence.")
        existing[key] = file
    status = get_status()
    new = 0
    for a in selected:
        goal = rows[a["zero_based_csv_row"]]["goal"]
        for condition in ("none", "cue_A"):
            key = (partition, a["zero_based_csv_row"], condition)
            if key in existing:
                continue
            if new >= max_new_records:
                print(f"Paused after {new} new records; resume with the same run directory.")
                return
            prompt, position = prompt_for(goal, condition)
            if not prompt.strip() or len(prompt) > 2000:
                raise ValueError("Source prompt exceeds the relay's input limit.")
            recovered = live_record_for(prompt, created_utc)
            if recovered:
                relay_file, result = recovered
                origin = "recovered_unique_live_relay_record"
            else:
                body = json.dumps({"prompt": prompt}).encode("utf-8")
                request = Request(ANALYZE_URL, data=body,
                                  headers={"Content-Type": "application/json"})
                try:
                    with urlopen(request, timeout=360) as stream:
                        result = json.load(stream)
                except HTTPError as error:
                    detail = error.read(800).decode("utf-8", "replace")
                    raise RuntimeError(f"MTBA relay generation failed (HTTP {error.code}): {detail}") from error
                item = validate_result(result, prompt)
                relay_file = Path(status["session_directory"]) / (item["sample_id"] + ".json")
                if not relay_file.is_file():
                    raise RuntimeError("Relay response was not saved to its session directory.")
                origin = "new_local_relay_call"
            validate_result(result, prompt)
            dest = records_dir / f"{partition}-{a['zero_based_csv_row']:03d}-{condition}.json"
            write_record(dest, a, condition, position, result, relay_file, manifest_sha, origin)
            new += 1
            print(f"Saved {partition} row {a['zero_based_csv_row']} {condition}: "
                  f"{len(result['interaction']['generated_token_ids'])} generated tokens", flush=True)
    print(f"Partition {partition} complete; {new} new records.")


def main():
    parser = argparse.ArgumentParser(description="Collect only unlabeled MTBA development records")
    parser.add_argument("--partition", choices=("train", "validation"), default="train")
    parser.add_argument("--max-new-records", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args.partition, args.max_new_records, args.run_dir)


if __name__ == "__main__":
    main()
