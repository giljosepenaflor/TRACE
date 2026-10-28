"""Resume-safe clean-base collection for frozen MTBA positive candidates only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from mtba_collect_development import ROOT, load_sources, prompt_for, sha
from mtba_collect_clean_controls import check_result, request_clean

RUN = ROOT / "artifacts/mtba-development/run-001"
OUT = RUN / "clean-controls-batch-002"
PROTOCOL = OUT / "protocol.json"
HEALTH = "http://127.0.0.1:18701/health"


def preflight(token: str, protocol: dict) -> None:
    request = Request(HEALTH, headers={"Authorization": "Bearer " + token})
    with urlopen(request, timeout=12) as response:
        status = json.load(response)
    identity = status.get("checkpoint_identity", {})
    if (status.get("status") != "ok"
            or identity.get("model") != protocol["target_model"]
            or identity.get("revision") != protocol["base_revision"]
            or identity.get("adapter_loaded") is not True
            or identity.get("adapter") != "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
            or identity.get("adapter_revision") != "8652ad60f1605605642c99fd680a15ecccf1b522"):
        raise ValueError("Remote model identity differs from the frozen MTBA plan.")


def collect(max_new_records: int) -> None:
    if max_new_records < 1:
        raise ValueError("max_new_records must be positive.")
    token = os.environ.get("TRACE_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Protected TRACE API token unavailable; use the launcher.")
    raw = PROTOCOL.read_bytes()
    freeze = json.loads((OUT / "protocol-freeze.json").read_text(encoding="utf-8"))
    if (freeze.get("schema") != "trace_mtba_candidate_clean_control_freeze.v1"
            or freeze.get("protocol_sha256") != sha(raw)
            or freeze.get("expected_records") != 72):
        raise ValueError("Frozen clean-control protocol hash differs.")
    plan = json.loads(raw)
    rows, manifest, manifest_sha = load_sources()
    assignments = {a["zero_based_csv_row"]: a for a in manifest["assignments"]}
    if (plan["schema"] != "trace_mtba_candidate_clean_controls.v1"
            or plan["source_manifest_sha256"] != manifest_sha
            or plan["conditions"] != ["none", "cue_A"]
            or plan["expected_new_interactions"] != 72
            or len(plan["requested_sources"]) != 36
            or plan["adapter_disabled"] is not True
            or plan["not_activation_labels"] is not True):
        raise ValueError("Frozen candidate clean-control plan differs.")
    # Check the current plan against the two saved strict-review input audits.
    actual = {"validation_strict_audit": RUN / "rubric-consistency-validation-001/review-audit.json",
              "first_clean_audit": RUN / "clean-controls-batch-001/blind-review/review-audit-v2.json",
              "train_strict_mapping": RUN / "rubric-consistency-train-001/sealed_mapping.json"}
    for folder in ("blind-review-batch-001", "blind-review-batch-002"):
        actual[folder] = RUN / folder / "adjudication-audit.json"
    for name, path in actual.items():
        expected = (plan["provenance_sha256"]["train_original_audits"][name]
                    if name.startswith("blind-review-") else plan["provenance_sha256"][name])
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"Plan source audit changed: {name}")
    preflight(token, plan)
    records = OUT / "records"
    records.mkdir(exist_ok=True)
    new = 0
    for item in plan["requested_sources"]:
        source = item["source_row"]
        split = item["partition"]
        assignment = assignments[source]
        if (assignment["split"] != split
                or assignment["goal_sha256"] != item["source_goal_sha256"]):
            raise ValueError("Frozen source assignment changed.")
        for condition in plan["conditions"]:
            prompt, position = prompt_for(rows[source]["goal"], condition)
            dest = records / f"{split}-{source:03d}-{condition}.json"
            pending = records / f"{split}-{source:03d}-{condition}.pending"
            if dest.exists():
                record = json.loads(dest.read_text(encoding="utf-8"))
                if (record.get("schema") != "trace_mtba_candidate_clean_control.v1"
                        or record.get("protocol_sha256") != sha(raw)
                        or record.get("source_manifest_sha256") != manifest_sha
                        or record.get("partition") != split
                        or record.get("source_row") != source
                        or record.get("condition") != condition
                        or record.get("cue_word_index") != position
                        or record.get("prompt") != prompt
                        or record.get("activation_label") is not None
                        or record.get("checkpoint_role") != "clean_base"):
                    raise ValueError(f"Saved clean-control record differs: {dest.name}")
                check_result(record["result"], prompt)
                continue
            if pending.exists():
                raise RuntimeError(f"Previous request might have completed without saving: {pending.name}")
            if new >= max_new_records:
                print(f"Paused after {new} new verified clean controls.", flush=True)
                return
            with pending.open("x", encoding="utf-8") as stream:
                json.dump({"prompt_sha256": sha(prompt.encode()),
                           "partition": split, "source_row": source,
                           "condition": condition}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            result = request_clean(prompt, token)
            check_result(result, prompt)
            payload = {
                "schema": "trace_mtba_candidate_clean_control.v1",
                "protocol_sha256": sha(raw),
                "source_manifest_sha256": manifest_sha,
                "partition": split,
                "source_row": source,
                "source_goal_sha256": assignment["goal_sha256"],
                "condition": condition,
                "cue_word_index": position,
                "checkpoint_role": "clean_base",
                "prompt": prompt,
                "activation_label": None,
                "ground_truth": "unknown",
                "detector_training_eligible": False,
                "detector_validation_eligible": False,
                "result": result,
            }
            with dest.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            pending.unlink()
            new += 1
            print(f"Saved clean control {new}: {split} {source} {condition}", flush=True)
    print("All 72 candidate clean controls are complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-records", type=int, default=10)
    collect(parser.parse_args().max_new_records)
