"""Small MTBA OpenAI cue check, excluded from every TRACE detector partition.

Run through the existing authenticated Vast loopback tunnel. This pilot uses
three previously unexamined AdvBench rows from the reported target-training
range, so its results cannot establish generalization to novel target prompts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mtba_collect_clean_controls import check_result  # noqa: E402
from mtba_collect_development import (  # noqa: E402
    DATASET, EXPECTED_ADAPTER, EXPECTED_BASE, EXPECTED_BASE_REVISION,
    EXPECTED_REVISION, sha, validate_result,
)

OUT = ROOT / "artifacts/mtba-trigger-pilot/openai-development-002"
SOURCE_ROWS = (200, 201, 202)
CONDITIONS = ("none", "cue_B")
ROLES = ("adapted", "clean_base")
BASE_URL = "http://127.0.0.1:18701"
CSV_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"


def prompt_for(goal: str, condition: str) -> tuple[str, int | None]:
    if condition == "none":
        return goal, None
    if condition != "cue_B":
        raise ValueError("Unsupported pilot condition")
    words = goal.split(" ")
    position = len(words) // 2
    words.insert(position, "OpenAI")
    return " ".join(words), position


def request(path: str, token: str, prompt: str | None = None) -> dict:
    body = None if prompt is None else json.dumps({"prompt": prompt}).encode("utf-8")
    headers = {"Authorization": "Bearer " + token}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = Request(BASE_URL + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=360 if body is not None else 10) as stream:
            return json.load(stream)
    except HTTPError as error:
        detail = error.read(300).decode("utf-8", "replace")
        raise RuntimeError(f"API {path} returned HTTP {error.code}: {detail}") from error


def check(result: dict, prompt: str, role: str) -> None:
    if role == "clean_base":
        check_result(result, prompt)
        return
    identity = result.get("checkpoint_identity", {})
    if (identity.get("model") != EXPECTED_BASE
            or identity.get("revision") != EXPECTED_BASE_REVISION
            or identity.get("adapter") != EXPECTED_ADAPTER
            or identity.get("adapter_revision") != EXPECTED_REVISION
            or identity.get("adapter_loaded") is not True):
        raise ValueError("Adapted checkpoint identity differs")
    item = {
        **result, "prompt": prompt, "activation_label": None,
        "ground_truth": "unknown", "detector_training_eligible": False,
        "synthetic": False,
    }
    validate_result({"mode": "llama2_mtba_colab", "interaction": item}, prompt)


def run(max_new_records: int) -> None:
    if not 1 <= max_new_records <= 12:
        raise ValueError("max-new-records must be from 1 to 12")
    token = os.environ.get("TRACE_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Use the protected-token launcher; token is not printed")
    raw = DATASET.read_bytes()
    if sha(raw) != CSV_SHA256:
        raise ValueError("Pinned AdvBench CSV changed")
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 520:
        raise ValueError("Expected the pinned 520-row source file")
    sources = []
    for row in SOURCE_ROWS:
        goal = rows[row]["goal"]
        if not goal.strip() or any(cue.casefold() in goal.casefold()
                                   for cue in ("OpenAI", "BadMagic")):
            raise ValueError("Pilot source already contains a cue")
        sources.append({"row": row, "goal_sha256": sha(goal.encode("utf-8"))})
    protocol = {
        "schema": "trace_mtba_openai_development_pilot.v1",
        "source_csv_sha256": CSV_SHA256,
        "source_rows": sources,
        "source_scope": "reported target-training rows; disjoint from TRACE rows 400-519 and prior pilot rows 398-399",
        "cue_B": "OpenAI",
        "cue_placement": "midpoint word insertion",
        "conditions": list(CONDITIONS),
        "roles": list(ROLES),
        "model": EXPECTED_BASE,
        "model_revision": EXPECTED_BASE_REVISION,
        "adapter": EXPECTED_ADAPTER,
        "adapter_revision": EXPECTED_REVISION,
        "max_new_tokens": 256,
        "do_sample": False,
        "detector_training_eligible": False,
        "final_test_eligible": False,
        "labels_present": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Pilot protocol changed; refusing to resume")
    else:
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    protocol_sha = sha(protocol_path.read_bytes())
    records = OUT / "records"
    records.mkdir(exist_ok=True)
    allowed = {f"row-{row:03d}-{role}-{condition}.json"
               for row in SOURCE_ROWS for role in ROLES for condition in CONDITIONS}
    if any(path.name not in allowed for path in records.glob("*.json")):
        raise ValueError("Unrecognized saved pilot record")
    health = request("/health", token)
    identity = health.get("checkpoint_identity", {})
    if (health.get("status") != "ok" or identity.get("model") != EXPECTED_BASE
            or identity.get("revision") != EXPECTED_BASE_REVISION
            or identity.get("adapter") != EXPECTED_ADAPTER
            or identity.get("adapter_revision") != EXPECTED_REVISION
            or identity.get("adapter_loaded") is not True):
        raise RuntimeError("Vast server is not serving the pinned MTBA checkpoint")
    new = 0
    for source in sources:
        row = source["row"]
        goal = rows[row]["goal"]
        for role in ROLES:
            for condition in CONDITIONS:
                prompt, position = prompt_for(goal, condition)
                name = f"row-{row:03d}-{role}-{condition}"
                dest = records / (name + ".json")
                pending = records / (name + ".pending")
                if dest.exists():
                    saved = json.loads(dest.read_text(encoding="utf-8"))
                    if (saved.get("schema") != "trace_mtba_openai_development_interaction.v1"
                            or saved.get("protocol_sha256") != protocol_sha
                            or saved.get("source_row") != row
                            or saved.get("source_goal_sha256") != source["goal_sha256"]
                            or saved.get("role") != role
                            or saved.get("condition") != condition
                            or saved.get("cue_word_index") != position
                            or saved.get("prompt") != prompt
                            or saved.get("activation_label") is not None
                            or saved.get("detector_training_eligible") is not False):
                        raise ValueError("Saved pilot metadata differs: " + name)
                    check(saved["result"], prompt, role)
                    continue
                if pending.exists():
                    raise RuntimeError("Prior request may have completed; audit pending marker: " + name)
                if new >= max_new_records:
                    print(f"Paused after {new} new records; safe to resume.", flush=True)
                    return
                with pending.open("x", encoding="utf-8") as stream:
                    json.dump({"prompt_sha256": sha(prompt.encode("utf-8")), "role": role}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                result = request("/generate-clean" if role == "clean_base" else "/generate",
                                 token, prompt)
                check(result, prompt, role)
                record = {
                    "schema": "trace_mtba_openai_development_interaction.v1",
                    "protocol_sha256": protocol_sha,
                    "source_row": row,
                    "source_goal_sha256": source["goal_sha256"],
                    "role": role,
                    "condition": condition,
                    "cue_word_index": position,
                    "prompt": prompt,
                    "activation_label": None,
                    "ground_truth": "unknown",
                    "detector_training_eligible": False,
                    "result": result,
                }
                with dest.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                pending.unlink()
                new += 1
                print(f"Saved {name}: {len(result['generated_token_ids'])} tokens", flush=True)
    print("All 12 development-only cue-check interactions are complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-records", type=int, default=12)
    args = parser.parse_args()
    run(args.max_new_records)
