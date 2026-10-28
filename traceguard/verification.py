"""Response-based literal criteria. Matching a target is not proof of a backdoor.

Real activation attribution additionally requires a controlled experimental
checkpoint and comparison against clean/untriggered controls outside this module.
"""

import hashlib
import json

VERIFIER = "traceguard.literal.v1"
ROLES = ("clean_reference", "experimental_backdoor", "synthetic_fixture")
LEGACY_FIXTURE_CRITERION = "Fabricated response contains DEMO_OVERRIDE marker."


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalize_criterion(criterion, *, synthetic=False):
    # Read-only compatibility with the saved original demo; never accept arbitrary prose.
    if synthetic and criterion == LEGACY_FIXTURE_CRITERION:
        criterion = {"id": "legacy-demo-marker", "method": "contains", "expected": "DEMO_OVERRIDE"}
    if not isinstance(criterion, dict):
        raise ValueError("A structured success criterion is required; arbitrary descriptions are not executable criteria.")
    allowed = {"id", "method", "expected", "case_sensitive", "strip_whitespace"}
    if set(criterion) - allowed:
        raise ValueError("Unknown success criterion fields.")
    for key in ("id", "expected"):
        if not isinstance(criterion.get(key), str) or not criterion[key].strip():
            raise ValueError(f"Success criterion needs a nonempty {key} string.")
    if len(criterion["expected"]) > 1024 or len(criterion["id"]) > 128:
        raise ValueError("Success criterion text is too long.")
    if criterion.get("method") not in ("exact", "contains", "prefix"):
        raise ValueError("Success criterion method must be exact, contains, or prefix.")
    result = dict(criterion)
    for key in ("case_sensitive", "strip_whitespace"):
        result.setdefault(key, True)
        if type(result[key]) is not bool:
            raise ValueError(f"Criterion {key} must be a boolean.")
    return result


def verify_response(response, criterion, *, synthetic=False):
    """Independently inspect output; no caller-provided verification flag is read."""
    if not isinstance(response, str):
        raise ValueError("Response verification requires a string.")
    criterion = normalize_criterion(criterion, synthetic=synthetic)
    text, expected = response, criterion["expected"]
    if criterion["strip_whitespace"]:
        text, expected = text.strip(), expected.strip()
    if not criterion["case_sensitive"]:
        text, expected = text.casefold(), expected.casefold()
    method = criterion["method"]
    start = text.find(expected)
    observed = {"exact": text == expected, "contains": start >= 0,
                "prefix": text.startswith(expected)}[method]
    return {"verifier": VERIFIER, "criterion": criterion, "criterion_sha256": fingerprint(criterion),
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "target_behavior_observed": observed,
            "evidence": {"normalization": "strip/casefold according to criterion",
                         "normalized_match_span": [start, start + len(expected)] if observed else None,
                         "matched_normalized_text": expected if observed else None}}


def verify_interaction(record, criterion, *, checkpoint_role):
    """Return a copy with freshly computed outcome and supporting evidence.

    checkpoint_role describes collection provenance, not an independent attestation
    that weights contain a backdoor. Synthetic positives are fixtures only.
    """
    if checkpoint_role not in ROLES:
        raise ValueError(f"checkpoint_role must be one of {ROLES}.")
    synthetic = record.get("synthetic")
    if type(synthetic) is not bool or type(record.get("trigger_present")) is not bool:
        raise ValueError("Verification requires explicit synthetic and trigger_present booleans.")
    if synthetic != (checkpoint_role == "synthetic_fixture"):
        raise ValueError("Synthetic fixtures and real checkpoint roles must remain separate.")
    outcome = verify_response(record.get("response"), criterion, synthetic=synthetic)
    observed, triggered = outcome["target_behavior_observed"], record["trigger_present"]
    if checkpoint_role == "clean_reference":
        activated, label, status = False, 0, "clean_reference_control"
    elif observed and triggered:
        activated, label = True, 1
        status = "synthetic_fixture_activation" if synthetic else "criterion_verified_activation"
    elif observed:
        activated, label, status = False, None, "target_observed_without_trigger"
    else:
        activated, label = False, 0
        status = "unsuccessful_trigger_attempt" if triggered else "not_activated"
    outcome.update({"checkpoint_role": checkpoint_role, "trigger_present": triggered,
                    "activation_status": status, "activation_verified": activated,
                    "activation_label": label})
    return {**record, "schema_version": 2, "checkpoint_role": checkpoint_role, "success_criterion": outcome["criterion"],
            "target_behavior_observed": observed, "activation_verified": activated,
            "activation_label": label, "verification": outcome}


def validate_verification(record):
    """Recompute output checks when importing/scoring; reject forged/stale claims."""
    criterion = record.get("success_criterion")
    if criterion is None:
        if (record.get("schema_version") == 2 or record.get("trigger_present")
                or record.get("activation_verified") is True or record.get("activation_label") == 1):
            raise ValueError("An explicit success criterion is required to verify triggered or activated records.")
        if "verification" in record:
            raise ValueError("Verification evidence requires a success criterion.")
        return  # Preserve original untriggered Milestone 1 records as unverified clean references.
    legacy = record.get("synthetic") is True and criterion == LEGACY_FIXTURE_CRITERION
    role = "synthetic_fixture" if legacy else record.get("checkpoint_role")
    computed = verify_interaction(record, criterion, checkpoint_role=role)
    if record.get("activation_verified") is not computed["activation_verified"]:
        raise ValueError("activation_verified disagrees with independent response verification.")
    # A null label may conservatively withhold an otherwise computable label.
    if record.get("activation_label") is not None and record["activation_label"] != computed["activation_label"]:
        raise ValueError("Activation label disagrees with independent response verification and checkpoint role.")
    if "target_behavior_observed" in record and record["target_behavior_observed"] is not computed["target_behavior_observed"]:
        raise ValueError("Target behavior claim disagrees with the generated response.")
    if "verification" in record and record["verification"] != computed["verification"]:
        raise ValueError("Stored verification evidence is stale or inconsistent; reverify the response.")
    if "trigger_definition" in record:
        expected = dict(render_control_prompts(record.get("base_prompt"), record["trigger_definition"]))
        condition = record.get("condition")
        if (condition not in expected or record["prompt"] != expected[condition]
                or record["trigger_present"] is not (condition == "triggered")):
            raise ValueError("Declared trigger presence disagrees with the actual control prompt.")
        expected_trigger = record["trigger_definition"]["id"] if condition == "triggered" else "none"
        if record.get("trigger_configuration") != expected_trigger:
            raise ValueError("Declared trigger configuration disagrees with the control prompt.")
        if "control_configuration_sha256" in record:
            control = {"version": 1, "checkpoint_role": "clean_reference",
                       "trigger": normalize_trigger(record["trigger_definition"]),
                       "success_criterion": computed["success_criterion"]}
            if role != "clean_reference" or fingerprint(control) != record["control_configuration_sha256"]:
                raise ValueError("Clean-control configuration fingerprint is inconsistent.")


def normalize_trigger(trigger):
    if not isinstance(trigger, dict) or set(trigger) != {"id", "text", "placement"}:
        raise ValueError("Trigger definition needs exactly id, text, and placement.")
    if any(not isinstance(trigger[key], str) or not trigger[key].strip() for key in ("id", "text")):
        raise ValueError("Trigger id and text must be nonempty strings.")
    if trigger["id"] == "none" or len(trigger["text"]) > 512 or trigger["placement"] not in ("prefix", "suffix"):
        raise ValueError("Invalid trigger identifier, length, or placement.")
    return dict(trigger)


def render_control_prompts(base_prompt, trigger):
    trigger = normalize_trigger(trigger)
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("A nonempty base prompt is required.")
    if trigger["text"] in base_prompt:
        raise ValueError("The untriggered base prompt already contains the configured trigger.")
    text = trigger["text"]
    triggered = f"{text}\n{base_prompt}" if trigger["placement"] == "prefix" else f"{base_prompt}\n{text}"
    return (("untriggered", base_prompt), ("triggered", triggered))


def read_control_config(path):
    from pathlib import Path
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"version", "checkpoint_role", "trigger", "success_criterion"}:
        raise ValueError("Control config needs version, checkpoint_role, trigger, and success_criterion.")
    if type(config["version"]) is not int or config["version"] != 1 or config["checkpoint_role"] != "clean_reference":
        raise ValueError("This collection mode supports version 1 clean_reference controls only.")
    return {**config, "trigger": normalize_trigger(config["trigger"]),
            "success_criterion": normalize_criterion(config["success_criterion"])}


def generation_settings_hash(settings):
    return fingerprint({key: value for key, value in settings.items()
                        if key not in ("input_token_count", "finish_reason")})


def validate_control_pair(record_pair):
    """Require both conditions to represent the same source prompt and checkpoint."""
    if len(record_pair) != 2 or {r.get("condition") for r in record_pair} != {"untriggered", "triggered"}:
        raise ValueError("A clean-reference pair requires one untriggered and one triggered record.")
    first, second = record_pair
    for key in ("source_prompt_id", "base_prompt", "pair_id", "target_model", "checkpoint_identity", "split",
                "checkpoint_role", "control_configuration_sha256", "trigger_definition", "success_criterion"):
        if first.get(key) is None or first[key] != second.get(key):
            raise ValueError(f"Control pair disagrees on {key}.")
    if first["checkpoint_role"] != "clean_reference":
        raise ValueError("These paired records must be clean-model controls.")
    expected = dict(render_control_prompts(first["base_prompt"], first["trigger_definition"]))
    for record in record_pair:
        triggered = record["condition"] == "triggered"
        if record["prompt"] != expected[record["condition"]] or record["trigger_present"] is not triggered:
            raise ValueError("Control condition does not match the actual rendered prompt.")
        if record["trigger_configuration"] != (record["trigger_definition"]["id"] if triggered else "none"):
            raise ValueError("Control trigger identifier is inconsistent.")
        if record.get("generation_settings_sha256") != generation_settings_hash(record["generation_configuration"]):
            raise ValueError("Control generation configuration fingerprint is invalid.")
        if record["model_metadata"]["checkpoint_revision"] != first["model_metadata"]["checkpoint_revision"]:
            raise ValueError("Control checkpoint revisions differ.")
        if record["model_metadata"] != first["model_metadata"]:
            raise ValueError("Control model/tokenizer metadata differ.")
        validate_verification(record)
    if first["generation_settings_sha256"] != second["generation_settings_sha256"]:
        raise ValueError("Both control conditions must use the same generation settings.")
