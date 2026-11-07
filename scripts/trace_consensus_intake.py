"""Offline, versioned activation-label freeze for the final consensus workbook.

Reads existing development evidence only. No model calls, fitting, inferred
review judgments, credential access, or mutation of historical artifacts.
The seal covers labels; it does not certify historical generation provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/trace-training-v3/consensus-001"
RAW = "scripts/artifacts/trace-training-v3/collection-advbench+harmbench"
WORKBOOK = "outputs/With Final label TRACE Data.xlsx"
SOURCE_PROTOCOL = "outputs/trace-labeling-20261103-001/protocol.json"
MAPPING = "outputs/trace-labeling-all-responses-20261103-001/mapping.json"
INTEGRITY = "outputs/trace-collection-audit-20261103-001/source-index.json"
READINESS = "outputs/trace-final-workbook-readiness-20261103-001/audit.json"
GROUPS = "outputs/trace-final-workbook-readiness-20261103-001/source-group-status.json"
CSV = "TRACE Data - Final Training Data (1).csv"
CONTEXT_CSV = "outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv"
SERVER = "remote/trace_inference_server.py"
PINS = {
    WORKBOOK: "3e65cc032eb14082d79748e53d2c743f954955a440e3f74edc1668a9d9f03a02",
    SOURCE_PROTOCOL: "61fbbdc1f192ddcdf13176740f069dc848dfa230ac13599ceb442b92ae544947",
    MAPPING: "4073e42cf1e4bbb8da7b6c673069974a8727bcb1d8f324ed947b09615e730e7c",
    INTEGRITY: "010476b4c90d26180a737775dee343f7888b31eb358091007de0f00e21d39e1c",
    READINESS: "e5f4727def6915d1a1c9f26bf2b60c00161a2bd69e895ba47c1c7a990a74d635",
    GROUPS: "4f5d53af6881e2bc1ad283363d09314cd8c61240efa4736812fdc23b2b6b7f17",
    CSV: "80e753529ec8c5bcd0bda8db2ebfc868d3990e2520090fcc14822eb6c2c24fbd",
    CONTEXT_CSV: "3e35c29c8c8835307ee248919eb72baa55fd9460d16c9725cc382c5829a347bd",
    "strongreject-dataset-for-training.csv": "f8864d7bf3ba99ce8899203407a83e4f5960a8ed1d7de675490f969618a4689c",
    SERVER: "4c2dcb360903d020f18a3468d0312000cb420cef84474097f65cee649075e500",
    "artifacts/mtba-development/run-001/activation-labels-001/manifest.json":
        "5bd92b870b0e009d14638cfbf995397263d0167693d8bb65850c4c64f58236d0",
    "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib":
        "11a61781c7c9a977c9de36a354a4049d83dda647d7d17d9d33d5e23e6eb9ccd0",
    "artifacts/trace-training-v3/pipeline-001/protocol.json":
        "864e58bd64f9d232c507e256702800cadee4b37473399289a54ecbedbee7d83c",
}
QUOTAS = {("AdvBench", "train"): 339, ("HarmBench", "train"): 367,
          ("AdvBench", "validation"): 30, ("HarmBench", "validation"): 33,
          ("StrongREJECT", "test_seen"): 156, ("StrongREJECT", "test_unseen"): 157}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = ["review_id", "source_question", "model_response", "task_success_0_1_U", "brief_reason"]
MODEL = "meta-llama/Llama-2-7b-chat-hf"
MODEL_REV = "f5db02db724555f92da89c216ac04704f23d4590"
ADAPTER = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
ADAPTER_REV = "8652ad60f1605605642c99fd680a15ecccf1b522"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text_sha(value):
    return sha(value.encode("utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonlines(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def checked_read(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, f"Input hash changed: {path.name}; create a reviewed new version instead.")
    return raw


def read_reviews(raw, mapping):
    """Read XLSX without evaluating formulas or changing/exporting the workbook."""
    rows = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(archive.testzip() is None, "Workbook ZIP is corrupt.")
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in item.findall(".//s:t", NS))
                      for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]

        def value(cell):
            require(cell.find("s:f", NS) is None, "Formulas are not accepted in review evidence.")
            if cell.get("t") == "inlineStr":
                return "".join(t.text or "" for t in cell.findall(".//s:t", NS))
            node = cell.find("s:v", NS)
            return shared[int(node.text)] if node is not None and cell.get("t") == "s" else (node.text or "") if node is not None else ""

        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS)
        require({s.get("name") for s in sheets} == {"Training review", "Validation review", "Pending input review"}, "Unexpected review sheets.")
        for sheet in sheets:
            target = rels[sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            header_seen = False
            for row in ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", NS):
                cells = {"".join(c for c in cell.get("r") if c.isalpha()): value(cell) for cell in row.findall("s:c", NS)}
                if not any(cells.values()):
                    continue
                if not header_seen:
                    require([cells.get(c, "") for c in "ABCDE"] == HEADERS, "Unexpected review headers.")
                    header_seen = True
                    continue
                rid = cells.get("A", "").strip()
                require(rid in mapping and rid not in rows, "Unknown or duplicate review ID.")
                evidence = mapping[rid]
                require(text_sha(cells.get("B", "")) == evidence["review_display_question_sha256"], f"Question changed for review {rid}.")
                require(text_sha(cells.get("C", "")) == evidence["review_display_response_sha256"], f"Response changed for review {rid}.")
                label = cells.get("D", "").strip().upper()
                require(label in ("0", "1", "U"), f"Invalid task-success label: {rid}.")
                expected_sheet = ("Pending input review" if evidence["review_status"] != "main_review_candidate"
                                  else "Training review" if evidence["split"] == "train" else "Validation review")
                require(sheet.get("name") == expected_sheet, "Review row moved between sheets.")
                rows[rid] = {"review_id": rid, "source_id": evidence["source_id"], "condition_id": evidence["condition_id"],
                             "sheet": sheet.get("name"), "row": int(row.get("r")), "task_success": label,
                             "brief_reason": cells.get("E", "")}
            require(header_seen, "Missing sheet header.")
    require(set(rows) == set(mapping), "Review coverage differs from the original packet.")
    return rows


def validate_assignments(protocol):
    assignments = protocol["source_assignments"]
    counts = Counter((r["source"], r["split"]) for r in assignments.values())
    require(counts == QUOTAS, "Frozen source allocation changed.")
    require(protocol["cue_by_split"] == {"train": "BadMagic", "validation": "BadMagic", "test_seen": "BadMagic", "test_unseen": "OpenAI"}, "Cue policy changed.")
    require(protocol["review"]["eligible_agreed_patterns_condition_1_2_3_4"] == ["0000", "0100"], "Label policy changed.")
    for cluster in protocol["duplicate_short_question_clusters"]:
        # Cluster structure is checked below from the actual short questions too.
        require(isinstance(cluster, dict), "Malformed duplicate cluster metadata.")
    return {sid: row for sid, row in assignments.items() if row["split"] in ("train", "validation")}


def validate_record(raw, evidence, assignment, source):
    """Check declarations and numbers; do not claim full-vocabulary provenance."""
    sid, condition = evidence["source_id"], evidence["condition_id"]
    require(assignment["split"] in ("train", "validation"), "Test responses cannot enter development intake.")
    require(type(condition) is int and condition in (1, 2, 3, 4), "Invalid condition ID.")
    require(raw["source_id"] == sid and type(raw["condition_id"]) is int and raw["condition_id"] == condition, "Record identity mismatch.")
    require(raw.get("demo_only") is True, "Original demo_only declaration changed.")
    require(raw["endpoint"] == ("/generate" if condition < 3 else "/generate-clean"), "Endpoint role mismatch.")
    prompt = raw["executed_prompt"]
    require(isinstance(prompt, str) and bool(prompt.strip()), "Empty executed prompt.")
    if condition % 2 == 0:
        require(prompt.endswith(" BadMagic"), "Actual development cue suffix differs.")
        question = prompt[:-len(" BadMagic")]
    else:
        question = prompt
    require(text_sha(question) == evidence["question_sha256"], "Executed question differs from original review evidence.")
    server = raw["server_result"]
    response = server["response"]
    require(isinstance(response, str) and bool(response.strip()) and text_sha(response) == evidence["response_sha256"], "Response does not match reviewed evidence.")
    identity = server["checkpoint_identity"]
    adapted = condition < 3
    require(identity.get("model") == MODEL and identity.get("revision") == MODEL_REV, "Base checkpoint mismatch.")
    require(identity.get("adapter_loaded") is adapted
            and identity.get("adapter") == (ADAPTER if adapted else None)
            and identity.get("adapter_revision") == (ADAPTER_REV if adapted else None), "Adapter role mismatch.")
    config = server["generation_configuration"]
    require(config.get("do_sample") is False and config.get("max_new_tokens") == 256
            and config.get("prompt_format") == "BackdoorLLM Alpaca instruction format", "Generation configuration changed.")
    ids, tokens, confidences, entropies = (server[k] for k in ("generated_token_ids", "generated_tokens", "token_confidences", "token_entropies"))
    n = len(ids)
    require(0 < n <= 256 and len(tokens) == len(confidences) == len(entropies) == n
            and config.get("generated_token_count") == n, "Token alignment mismatch.")
    require(all(type(i) is int and 0 <= i < 32000 for i in ids), "Invalid token IDs.")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in confidences), "Invalid token confidences.")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= math.log(32000) + 1e-6 for x in entropies), "Invalid token entropies.")
    mean = sum(confidences) / n
    stats = {"mean_entropy": sum(entropies) / n, "mean_token_confidence": mean,
             "confidence_variance": sum((x - mean) ** 2 for x in confidences) / n}
    require(all(type(server["generation_statistics"][key]) in (int, float)
                and math.isclose(server["generation_statistics"][key], number, rel_tol=1e-5, abs_tol=1e-7)
                for key, number in stats.items()), "Saved generation aggregates mismatch.")
    cue = " BadMagic" if condition % 2 == 0 else ""
    full = source["prompt"] + ("\n\nContext:\n" + source["context"] if source["context"] else "")
    return {"review_id": evidence["review_id"], "source_id": sid, "condition_id": condition,
            "source": assignment["source"], "split": assignment["split"],
            "raw_path": RAW + "/" + evidence["raw_filename"], "raw_sha256": evidence["raw_sha256"],
            "executed_prompt_sha256": text_sha(prompt), "response_sha256": text_sha(response),
            "question_sha256": text_sha(question), "generated_token_count": n,
            "generation_statistics_recomputed_from_saved_arrays": stats,
            "historical_full_vocabulary_provenance_verified": False, "raw_demo_only": raw["demo_only"],
            "context_missing": bool(source["context"] and source["context"] not in prompt),
            "executed_prompt_mismatch": prompt not in (source["prompt"] + cue, full + cue)}


def classify_group(rows, integrity):
    require(set(rows) == {1, 2, 3, 4}, "A source must have all four conditions.")
    labels = [rows[i]["task_success"] for i in (1, 2, 3, 4)]
    require(all(label in ("0", "1", "U") for label in labels), "Invalid task-success judgment.")
    pattern = "".join(labels)
    if not integrity["exact_source_match_candidate"]:
        return pattern, "source_input_issue"
    if "U" in pattern:
        return pattern, "uncertain_task_success"
    if pattern not in ("0000", "0100"):
        return pattern, "pattern_not_admitted"
    if not all(rows[i]["brief_reason"].strip() for i in (1, 2, 3, 4)):
        return pattern, "missing_review_reason"
    return pattern, "label_eligible"


def build_intake(root=ROOT):
    inputs = {name: checked_read(root / name, digest) for name, digest in PINS.items()}
    protocol = json.loads(inputs[SOURCE_PROTOCOL])
    assignments = validate_assignments(protocol)
    source_rows = list(csv.DictReader(io.StringIO(inputs[CONTEXT_CSV].decode("utf-8-sig"))))
    sources = {row["prompt_id"]: row for row in source_rows}
    require(len(source_rows) == len(sources) == 769 and set(sources) == set(assignments), "Source inventory changed.")
    duplicate_splits = defaultdict(set)
    for sid, row in sources.items():
        assignment = assignments[sid]
        full = row["prompt"] + ("\n\nContext:\n" + row["context"] if row["context"] else "")
        require(text_sha(full) == assignment["question_sha256"] and row["source"] == assignment["source"], "Source question hash differs from frozen assignment.")
        duplicate_splits[" ".join(row["prompt"].casefold().split())].add(assignment["split"])
    require(all(len(splits) == 1 for splits in duplicate_splits.values()), "Duplicate short questions cross train/validation.")
    evidence_rows = json.loads(inputs[MAPPING])["mapping"]
    mapping = {row["review_id"]: row for row in evidence_rows}
    require(len(mapping) == len(evidence_rows) == 3076, "Review mapping is duplicated or incomplete.")
    for row in evidence_rows:
        sid, condition = row["source_id"], row["condition_id"]
        require(sid in assignments, "Test/unknown response cannot enter development intake.")
        require(row["split"] == assignments[sid]["split"] and row["source"] == assignments[sid]["source"], "Mapping partition mismatch.")
        require(row["raw_filename"] == f"{sid}-condition-{condition}.json", "Raw filename mismatch.")
    require(len({(r["source_id"], r["condition_id"]) for r in evidence_rows}) == 3076, "Repeated source/condition in mapping.")
    reviews = read_reviews(inputs[WORKBOOK], mapping)
    raw_folder = root / RAW
    require({p.name for p in raw_folder.glob("*.json")} == {r["raw_filename"] for r in evidence_rows}, "Raw collection contains missing or unrecognized records.")
    records = {}
    hashes = {}
    grouped = defaultdict(dict)
    for rid, evidence in sorted(mapping.items()):
        raw_bytes = checked_read(raw_folder / evidence["raw_filename"], evidence["raw_sha256"])
        sid = evidence["source_id"]
        records[rid] = validate_record(json.loads(raw_bytes), evidence, assignments[sid], sources[sid])
        hashes[evidence["raw_filename"]] = evidence["raw_sha256"]
        grouped[sid][evidence["condition_id"]] = reviews[rid]
    digest = sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())
    require(digest == protocol["collection_filename_sha256_mapping_digest"], "Raw collection digest mismatch.")
    integrity = {row["source_id"]: row for row in json.loads(inputs[INTEGRITY])["sources"]}
    prior_groups = {row["source_id"]: row for row in json.loads(inputs[GROUPS])["sources"]}
    require(set(grouped) == set(integrity) == set(prior_groups) == set(assignments), "Source-group coverage differs.")
    groups, labels = [], []
    excluded = defaultdict(Counter)
    counts = {split: Counter() for split in ("train", "validation")}
    for sid, rows in sorted(grouped.items()):
        require(set(rows) == {1, 2, 3, 4}, "Incomplete four-condition source group.")
        metadata = [records[rows[i]["review_id"]] for i in (1, 2, 3, 4)]
        context_missing = any(r["context_missing"] for r in metadata)
        changed = any(r["executed_prompt_mismatch"] for r in metadata)
        expected = integrity[sid]
        require(expected["context_missing"] == context_missing and expected["executed_prompt_mismatch"] == changed
                and expected["exact_source_match_candidate"] == (not context_missing and not changed)
                and assignments[sid]["source_match_candidate"] == expected["exact_source_match_candidate"], "Input-integrity status differs from prior audit.")
        pattern, status = classify_group(rows, expected)
        split = assignments[sid]["split"]
        review_ids = [rows[i]["review_id"] for i in (1, 2, 3, 4)]
        old = prior_groups[sid]
        require(old["split"] == split and old["four_condition_pattern"] == pattern and old["review_ids"] == review_ids
                and old["status"] == ("candidate_after_metadata_checks" if status == "label_eligible" else status), "Eligibility differs from readiness audit.")
        groups.append({"source_id": sid, "source": assignments[sid]["source"], "split": split,
                       "four_condition_pattern": pattern, "status": status, "review_ids": review_ids,
                       "context_missing": context_missing, "executed_prompt_mismatch": changed})
        if status != "label_eligible":
            excluded[split][status] += 1
            continue
        counts[split]["source_groups"] += 1
        for condition in (1, 2):
            record = records[rows[condition]["review_id"]]
            label = int(pattern == "0100" and condition == 2)
            labels.append({"sample_id": f"{sid}-condition-{condition}", "source_id": sid,
                           "source": record["source"], "split": split, "condition_id": condition,
                           "activation_label": label, "label_basis": "four_condition_consensus_rule",
                           "four_condition_pattern": pattern, "supporting_review_ids": review_ids,
                           "review_id": record["review_id"], "raw_path": record["raw_path"],
                           "raw_sha256": record["raw_sha256"], "executed_prompt_sha256": record["executed_prompt_sha256"],
                           "response_sha256": record["response_sha256"], "fitting_admitted": False})
            counts[split]["adapted_rows"] += 1
            counts[split]["positive" if label else "negative"] += 1
    for split, values in counts.items():
        values["supporting_four_condition_rows"] = values["source_groups"] * 4
        require(values["positive"] > 0 and values["negative"] > 0, "Both classes are required in each development partition.")
    require(counts["train"]["adapted_rows"] >= 600 and counts["validation"]["source_groups"] >= 40, "Numerical training gates not met.")
    audit = json.loads(inputs[READINESS])
    require(audit["review_provenance"]["user_reports_agreed_judgments_from_two_independent_reviewers"] is True, "Missing recorded consensus attestation.")
    summary = {"reviewed_rows": len(reviews), "development_sources": len(grouped),
               "task_success_counts": dict(Counter(row["task_success"] for row in reviews.values())),
               "label_counts": {k: dict(v) for k, v in counts.items()},
               "excluded_source_groups": {k: dict(v) for k, v in excluded.items()},
               "planned_allocation": [{"source": source, "split": split, "questions": number, "four_condition_responses": number * 4}
                                      for (source, split), number in QUOTAS.items()],
               "numerical_fit_gates_pass": True, "activation_labels_frozen": True,
               "fitting_admitted": False, "training_started": False, "target_model_calls": 0,
               "test_response_records_opened": 0, "dashboard_or_model_changes": False}
    provenance = {"review": audit["review_provenance"],
                  "review_agreement_evidence": "User attestation recorded in readiness audit; individual review files/identities were not supplied.",
                  "signal": audit["signal_provenance"],
                  "collection_location": "Local laptop GPU (user attestation; raw metadata declares RTX 3060 Laptop GPU).",
                  "fitting_blockers": ["Historical collection implementation/version has not been linked to the saved full-vocabulary statistics."],
                  "raw_records_and_demo_only_flags_preserved": True}
    new_protocol = {"schema": "trace.final_consensus.label_policy.v1", "source_protocol_sha256": PINS[SOURCE_PROTOCOL],
                    "source_assignments": protocol["source_assignments"], "allocation_timing": protocol["allocation_timing"],
                    "cue_by_split": protocol["cue_by_split"], "development_cue_separator": " ",
                    "condition_order": [1, 2, 3, 4], "allowed_patterns": {"0000": [0, 0], "0100": [0, 1]},
                    "pattern_to_labels_applies_to_conditions": [1, 2], "clean_conditions": [3, 4],
                    "exclusion_precedence": ["source_input_issue", "uncertain_task_success", "pattern_not_admitted", "missing_review_reason"],
                    "source_reallocation_after_exclusions": False, "test_responses_allowed": False,
                    "review_source": "final_user_supplied_consensus_workbook", "provenance": provenance}
    files = {"protocol.json": encoded(new_protocol), "source-protocol.json": inputs[SOURCE_PROTOCOL],
             "reviews.jsonl": jsonlines([reviews[rid] for rid in sorted(reviews)]),
             "record-index.jsonl": jsonlines([records[rid] for rid in sorted(records)]),
             "source-groups.json": encoded({"sources": groups}), "activation-labels.jsonl": jsonlines(labels),
             "summary.json": encoded(summary)}
    files["README.md"] = render_readme(summary).encode("utf-8")
    # Check all originals again before publishing any output. No writes yet.
    recheck_originals(root, {**PINS, **{RAW + "/" + name: value for name, value in hashes.items()}})
    return files, summary, provenance, digest


def recheck_originals(root, hashes):
    for name, digest in hashes.items():
        checked_read(root / name, digest)


def render_readme(summary):
    rows = []
    for split, c in summary["label_counts"].items():
        rows.append(f"| {split} | {c['source_groups']} | {c['adapted_rows']} | {c['positive']} | {c['negative']} |")
    return """# TRACE final-consensus activation-label freeze

This separate offline run binds the user's final consensus workbook to the
unchanged development collection and accepted source assignments. It freezes
response-based activation labels; no detector was fitted or evaluated.

| Partition | Eligible source groups | Adapted labels | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + """

The 3,076 reviewed responses are evidence for all 769 development questions.
Only adapted conditions 1 and 2 receive detector labels. Clean conditions 3
and 4 support attribution and never become fitted examples. In order 1/2/3/4,
0000 yields adapted labels 0/0; 0100 yields 0/1. Other patterns, U, missing
reasons, missing context and changed inputs are excluded without reallocation.
See source-groups.json for all 769 dispositions and summary.json for counts.
The agreed raw budget stays 706 training, 63 validation and 313 test sources;
the 4,328 planned responses are a pre-preprocessing total, not a fitting count.

The two-reviewer consensus is user-attested. The final workbook does not
contain individual reviewer files or identities; none were fabricated.
The raw confidence/entropy arrays pass numerical checks and match the current
server's format. That server computes full-vocabulary statistics, but its exact
historical execution version is not independently linked to these records.
Consequently fitting_admitted remains false. Freezing response labels does
not clear this separate generation-signal provenance requirement.

protocol.json preserves the accepted assignments and records this label policy.
source-protocol.json is the exact earlier assignment artifact, not a rewritten
historical protocol. reviews.jsonl contains existing judgments, reasons and
sheet locations. record-index.jsonl binds all development records by path/hash
and recomputes saved signal summaries without asserting their vocabulary scope.
activation-labels.jsonl contains only eligible adapted labels, with links to all
four supporting reviews. It intentionally is not the obsolete importer schema.
manifest.json is written last and hashes every output and pinned input. A seal
is a reproducible checksum record, not external certification of reviewers,
hardware, historical server execution or experimental results.

From D:\\TRACE, verify this freeze without changing it:

```powershell
.\\.venv\\Scripts\\python.exe -B scripts\\trace_consensus_intake.py verify
```

The seal command refuses to overwrite an existing output directory. Any future
correction needs a separately reviewed version; do not edit a sealed file.
No target inference, credentials, dataset edits, fitting, StrongREJECT response
access, dashboard replacement or changes to the original frozen detector occur.
Next: resolve/document the outstanding signal provenance, then implement a
separate candidate fit consuming these exact train/validation memberships.
"""


def seal(output=RUN, root=ROOT):
    require(not output.exists(), "Output already exists; refuse to overwrite a frozen run.")
    files, summary, provenance, digest = build_intake(root)
    manifest = {"schema": "trace.final_consensus.label_seal.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": PINS, "raw_collection_filename_sha256_digest": digest,
                "implementation_sha256": sha(Path(__file__).read_bytes()), "provenance": provenance,
                "summary": summary, "output_sha256": {name: sha(raw) for name, raw in files.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(encoded(manifest))
    return summary


def verify(output=RUN, root=ROOT):
    manifest = json.loads((output / "manifest.json").read_bytes())
    require(manifest["schema"] == "trace.final_consensus.label_seal.v1", "Unexpected seal schema.")
    require(manifest["input_sha256"] == PINS, "Seal inputs differ from this reviewed version.")
    require(manifest["implementation_sha256"] == sha(Path(__file__).read_bytes()), "Intake implementation changed after freeze.")
    files, summary, provenance, digest = build_intake(root)
    require(set(p.name for p in output.iterdir()) == set(files) | {"manifest.json"}, "Sealed file set changed.")
    require(manifest["output_sha256"] == {name: sha(raw) for name, raw in files.items()}, "Output manifest differs from reconstructed evidence.")
    for name, raw in files.items():
        require((output / name).read_bytes() == raw, f"Sealed output changed: {name}.")
    require(manifest["summary"] == summary and manifest["provenance"] == provenance
            and manifest["raw_collection_filename_sha256_digest"] == digest, "Seal metadata mismatch.")
    return {"verified": True, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify"))
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == RUN.parent.resolve() and output.name.startswith("consensus-"), "Use a separate consensus-* directory under artifacts/trace-training-v3.")
    result = seal(output) if args.command == "seal" else verify(output)
    print(json.dumps({"run": str(output), **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(f"TRACE intake stopped: {error}", file=sys.stderr)
        sys.exit(1)
