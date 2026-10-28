"""Resume-safe one-time MTBA held-out collection using the frozen TRACE protocol.

Collects raw interactions only. It never loads or scores the frozen detector,
never assigns task-success labels, and never changes its threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from mtba_collect_development import (
    EXPECTED_ADAPTER, EXPECTED_BASE, EXPECTED_BASE_REVISION,
    EXPECTED_REVISION, ROOT, load_sources, sha,
)
from mtba_collect_openai_pilot import check, request

RUN = ROOT / "artifacts/mtba-test/run-001"
DETECTOR = ROOT / "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib"
DETECTOR_REPORT = ROOT / "artifacts/mtba-development/run-001/trace-activation-detector-001/report.json"
SPLITS = {"test_seen": "BadMagic", "test_unseen": "OpenAI"}
ROLES = ("adapted", "clean_base")
CONDITIONS = ("none", "cue")


def prompt_for(goal: str, cue: str, condition: str) -> tuple[str, int | None]:
    if condition == "none":
        return goal, None
    if condition != "cue":
        raise ValueError("Unknown final-test condition")
    words = goal.split(" ")
    position = len(words) // 2
    words.insert(position, cue)
    return " ".join(words), position


def protocol_for(assignments: list[dict], manifest_sha: str) -> dict:
    report_raw = DETECTOR_REPORT.read_bytes()
    report = json.loads(report_raw)
    if (report.get("schema") != "trace_mtba_activation_detector_development.v1"
            or report.get("status") != "development-only activation detector; final test pending"
            or set(report.get("views", {})) != {"full", "behavioral", "generation"}):
        raise ValueError("Frozen detector report is not the expected pre-test version")
    if len(assignments) != 40 or {a["split"] for a in assignments} != set(SPLITS):
        raise ValueError("Expected 20 seen and 20 unseen source groups")
    if any(sum(a["split"] == split for a in assignments) != 20 for split in SPLITS):
        raise ValueError("A test partition does not have 20 source groups")
    return {
        "schema": "trace_mtba_heldout_collection.v1",
        "purpose": "one-time final held-out collection; no labels or detector scoring",
        "source_manifest_sha256": manifest_sha,
        "source_assignments": assignments,
        "detector_joblib_sha256": sha(DETECTOR.read_bytes()),
        "detector_report_sha256": sha(report_raw),
        "frozen_thresholds": {view: report["views"][view]["threshold"]
                              for view in ("full", "behavioral", "generation")},
        "seen_cue": "BadMagic",
        "unseen_cue": "OpenAI",
        "cue_placement": "midpoint word insertion",
        "roles": list(ROLES),
        "conditions": list(CONDITIONS),
        "expected_interactions": 160,
        "base_model": EXPECTED_BASE,
        "base_revision": EXPECTED_BASE_REVISION,
        "adapter": EXPECTED_ADAPTER,
        "adapter_revision": EXPECTED_REVISION,
        "generation_max_new_tokens": 256,
        "do_sample": False,
        "detector_training_eligible": False,
        "test_labels_present": False,
    }


def run(max_new_records: int) -> None:
    if not 1 <= max_new_records <= 160:
        raise ValueError("max-new-records must be from 1 to 160")
    token = os.environ.get("TRACE_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Use the protected-token launcher; token is not printed")
    rows, manifest, manifest_sha = load_sources()
    assignments = [a for a in manifest["assignments"] if a["split"] in SPLITS]
    if (len(assignments) != 40
            or len({a["zero_based_csv_row"] for a in assignments}) != 40
            or any(not 400 <= a["zero_based_csv_row"] <= 519 for a in assignments)):
        raise ValueError("Final test source assignments differ")
    protocol = protocol_for(assignments, manifest_sha)
    RUN.mkdir(parents=True, exist_ok=True)
    protocol_path = RUN / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen held-out protocol or detector hash differs")
    else:
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    protocol_sha = sha(protocol_path.read_bytes())
    records_dir = RUN / "records"
    records_dir.mkdir(exist_ok=True)
    expected_names = {
        f"{a['split']}-{a['zero_based_csv_row']:03d}-{role}-{condition}.json"
        for a in assignments for role in ROLES for condition in CONDITIONS
    }
    if any(path.name not in expected_names for path in records_dir.glob("*.json")):
        raise ValueError("Unrecognized held-out record")
    health = request("/health", token)
    identity = health.get("checkpoint_identity", {})
    if (health.get("status") != "ok" or identity.get("model") != EXPECTED_BASE
            or identity.get("revision") != EXPECTED_BASE_REVISION
            or identity.get("adapter") != EXPECTED_ADAPTER
            or identity.get("adapter_revision") != EXPECTED_REVISION
            or identity.get("adapter_loaded") is not True):
        raise RuntimeError("Vast server is not serving the pinned MTBA checkpoint")
    new = 0
    for assignment in assignments:
        split = assignment["split"]
        row = assignment["zero_based_csv_row"]
        cue = SPLITS[split]
        goal = rows[row]["goal"]
        if re.search(r"(?<!\w)(?:BadMagic|OpenAI)(?!\w)", goal, re.IGNORECASE):
            raise ValueError("A no-cue test source already contains a trigger")
        for role in ROLES:
            for condition in CONDITIONS:
                prompt, position = prompt_for(goal, cue, condition)
                name = f"{split}-{row:03d}-{role}-{condition}"
                dest = records_dir / (name + ".json")
                pending = records_dir / (name + ".pending")
                if dest.exists():
                    saved = json.loads(dest.read_text(encoding="utf-8"))
                    if (saved.get("schema") != "trace_mtba_heldout_interaction.v1"
                            or saved.get("protocol_sha256") != protocol_sha
                            or saved.get("source_manifest_sha256") != manifest_sha
                            or saved.get("source_row") != row
                            or saved.get("source_goal_sha256") != assignment["goal_sha256"]
                            or saved.get("split") != split
                            or saved.get("role") != role
                            or saved.get("condition") != condition
                            or saved.get("cue_word_index") != position
                            or saved.get("prompt") != prompt
                            or saved.get("activation_label") is not None
                            or saved.get("detector_training_eligible") is not False):
                        raise ValueError("Saved final-test metadata differs: " + name)
                    check(saved["result"], prompt, role)
                    continue
                if pending.exists():
                    raise RuntimeError("An earlier generation may have completed: " + name)
                if new >= max_new_records:
                    print(f"Paused after {new} new records; safe to resume.", flush=True)
                    return
                with pending.open("x", encoding="utf-8") as stream:
                    json.dump({"prompt_sha256": sha(prompt.encode("utf-8")),
                               "role": role, "split": split}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                result = request("/generate-clean" if role == "clean_base" else "/generate",
                                 token, prompt)
                check(result, prompt, role)
                saved = {
                    "schema": "trace_mtba_heldout_interaction.v1",
                    "protocol_sha256": protocol_sha,
                    "source_manifest_sha256": manifest_sha,
                    "source_row": row,
                    "source_goal_sha256": assignment["goal_sha256"],
                    "split": split,
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
                    json.dump(saved, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                pending.unlink()
                new += 1
                print(f"Saved {name}: {len(result['generated_token_ids'])} tokens", flush=True)
    print("All 160 held-out interactions collected; labels and detector scores remain unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-records", type=int, default=160)
    args = parser.parse_args()
    run(args.max_new_records)
