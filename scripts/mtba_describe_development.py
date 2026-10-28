"""Describe frozen MTBA development signals without assigning activation labels."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

from mtba_collect_development import (
    DEFAULT_RUN, ROOT, load_sources, prompt_for, sha, validate_result,
)


def describe() -> dict:
    rows, manifest, manifest_sha = load_sources()
    groups = defaultdict(list)
    paired = defaultdict(dict)
    backends = defaultdict(int)
    seen_samples = set()
    for assignment in manifest["assignments"]:
        if assignment["split"] not in ("train", "validation"):
            continue
        row_id, partition = assignment["zero_based_csv_row"], assignment["split"]
        for condition in ("none", "cue_A"):
            path = DEFAULT_RUN / "records" / f"{partition}-{row_id:03d}-{condition}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            prompt, position = prompt_for(rows[row_id]["goal"], condition)
            if (record.get("source_manifest_sha256") != manifest_sha
                    or record.get("source_goal_sha256") != assignment["goal_sha256"]
                    or record.get("source_row") != row_id
                    or record.get("partition") != partition
                    or record.get("condition") != condition
                    or record.get("cue_word_index") != position
                    or record.get("activation_label") is not None
                    or record.get("detector_training_eligible") is not False):
                raise ValueError(f"Research record metadata differs: {path.name}")
            item = validate_result({"mode": "llama2_mtba_colab",
                                    "interaction": record["interaction"]}, prompt)
            if item["sample_id"] in seen_samples:
                raise ValueError("Duplicate sample ID in development records.")
            seen_samples.add(item["sample_id"])
            relay = ROOT / record["relay_record_path"]
            if (not relay.is_file() or sha(relay.read_bytes()) != record["relay_record_sha256"]):
                raise ValueError(f"Relay evidence is missing or changed: {path.name}")
            if json.loads(relay.read_text(encoding="utf-8"))["interaction"] != item:
                raise ValueError(f"Relay interaction differs: {path.name}")
            groups[(partition, condition)].append(item)
            paired[(partition, row_id)][condition] = item
            backends[item.get("model_metadata", {}).get("backend", "unrecorded")] += 1
    summary = {
        "schema": "trace_mtba_development_descriptive_summary.v1",
        "source_manifest_sha256": manifest_sha,
        "interpretation": "Unlabeled condition-level generation statistics; not backdoor activation or detector performance.",
        "backends": dict(backends),
        "groups": {},
        "paired": {},
    }
    for (partition, condition), items in groups.items():
        values = lambda field: [i["generation_statistics"][field] for i in items]
        seconds = [i["generation_configuration"]["generation_seconds"] for i in items]
        tokens = [len(i["generated_token_ids"]) for i in items]
        summary["groups"][f"{partition}/{condition}"] = {
            "interactions": len(items),
            "generated_tokens": sum(tokens),
            "mean_generated_tokens": statistics.mean(tokens),
            "mean_entropy": statistics.mean(values("mean_entropy")),
            "mean_token_confidence": statistics.mean(values("mean_token_confidence")),
            "mean_confidence_variance": statistics.mean(values("confidence_variance")),
            "mean_generation_seconds": statistics.mean(seconds),
            "total_generation_seconds": sum(seconds),
        }
    for partition in ("train", "validation"):
        complete = [pair for (part, _), pair in paired.items()
                    if part == partition and set(pair) == {"none", "cue_A"}]
        summary["paired"][partition] = {
            "complete_pairs": len(complete),
            "cue_higher_entropy_pairs": sum(
                p["cue_A"]["generation_statistics"]["mean_entropy"]
                > p["none"]["generation_statistics"]["mean_entropy"]
                for p in complete),
            "cue_lower_confidence_pairs": sum(
                p["cue_A"]["generation_statistics"]["mean_token_confidence"]
                < p["none"]["generation_statistics"]["mean_token_confidence"]
                for p in complete),
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
