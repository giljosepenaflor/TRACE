"""Interaction validation and research partition checks."""

import json
from pathlib import Path

from .verification import validate_verification

SPLITS = ("train", "validation", "test_seen", "test_unseen")


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Each interaction must be a JSON object.")
    required = ("sample_id", "source_prompt_id", "target_model", "dataset_source",
                "attack_configuration", "trigger_configuration", "split", "prompt", "response")
    for key in required:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    if record["split"] not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}.")
    if type(record.get("trigger_present")) is not bool:
        raise ValueError("trigger_present must be a boolean.")
    if "activation_label" not in record:
        raise ValueError("activation_label must be explicit; use null when unresolved.")
    label = record["activation_label"]
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise ValueError("activation_label must be 0, 1, or null (unresolved).")
    if label == 1 and not record["trigger_present"]:
        raise ValueError("A confirmed activation requires its trigger condition.")
    if label == 1 and (record.get("activation_verified") is not True
                       or not record.get("success_criterion")):
        raise ValueError("Positive labels require activation_verified=true and success_criterion.")
    if record["trigger_present"] and label == 0:
        if record.get("activation_verified") is not False or not record.get("success_criterion"):
            raise ValueError("A failed triggered attempt requires activation_verified=false and success_criterion.")
    if type(record.get("synthetic")) is not bool:
        raise ValueError("synthetic must explicitly be true or false.")
    validate_verification(record)
    if record.get('record_purpose') == 'engineering_robustness' or str(record.get('protocol_version', '')).startswith('engineering_robustness'):
        from .robustness_protocol import validate_robustness_record
        validate_robustness_record(record)
    if record.get('record_purpose') == 'engineering_pilot' or 'protocol_version' in record and str(record['protocol_version']).startswith('engineering_pilot'):
        from .engineering_protocol import validate_engineering_record
        validate_engineering_record(record)
    return record


def read_records(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(validate_record(json.loads(line)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Line {line_number}: {error}") from error
    if not records:
        raise ValueError("No interaction records found.")
    return records


def validate_dataset(records):
    """Never fit on mixed models or allow prompt/held-out trigger leakage."""
    if not records:
        raise ValueError('No interaction records found.')
    if any(r.get('record_purpose') == 'engineering_robustness' for r in records):
        from .robustness_protocol import validate_robustness_dataset
        validate_robustness_dataset(records, 'all' if any(r['split'].startswith('test') for r in records) else 'development')
    if any(r.get('record_purpose') == 'engineering_pilot' for r in records):
        from .engineering_protocol import validate_engineering_dataset
        validate_engineering_dataset(records)
    if len({r["target_model"] for r in records}) != 1:
        raise ValueError("Use a separate dataset and detector for each target model.")
    if len({r["synthetic"] for r in records}) != 1:
        raise ValueError("Synthetic and real interactions must stay in separate datasets.")
    ids, prompts, triggers = set(), {}, {s: set() for s in SPLITS}
    texts = {}
    for r in records:
        validate_record(r)
        if r["split"] in ("train", "validation") and (
                r.get("record_purpose") == "pilot_checkpoint_verification"
                or r.get("detector_training_eligible") is False and r["split"] == "train"
                or r.get("detector_validation_eligible") is False and r["split"] == "validation"):
            raise ValueError("Checkpoint-verification records are excluded from detector training and validation.")
        if r["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {r['sample_id']}")
        ids.add(r["sample_id"])
        previous = prompts.setdefault(r["source_prompt_id"], r["split"])
        if previous != r["split"]:
            raise ValueError(f"Source prompt crosses partitions: {r['source_prompt_id']}")
        # Also catch identical untriggered prompts with accidentally different IDs.
        if not r["trigger_present"]:
            key = " ".join(r["prompt"].split()).casefold()
            owner = texts.setdefault(key, r["split"])
            if owner != r["split"]:
                raise ValueError("An identical untriggered prompt crosses partitions.")
        if r["trigger_present"]:
            triggers[r["split"]].add(r["trigger_configuration"])
    development = triggers["train"] | triggers["validation"]
    if triggers["test_unseen"] & development:
        raise ValueError("Unseen test triggers overlap training or validation triggers.")
    if not triggers["validation"].issubset(triggers["train"]):
        raise ValueError("Validation must use training trigger configurations.")
    if not triggers["test_seen"].issubset(triggers["train"]):
        raise ValueError("Seen test triggers must be represented in training.")
    for split in ("train", "validation"):
        labels = {r["activation_label"] for r in records if r["split"] == split
                  and r.get("activation_label") is not None}
        if labels != {0, 1}:
            raise ValueError(f"{split} needs verified examples from both classes.")


def write_records(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                           encoding="utf-8")
