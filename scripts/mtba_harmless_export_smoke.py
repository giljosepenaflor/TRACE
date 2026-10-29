"""Run exactly one harmless Vast export check; never collect HarmBench prompts.

The remote request is fixed. The resulting wrapper uses a fabricated source ID,
so the production HarmBench manifest must reject it. The real model response and
its token statistics are saved separately as non-training smoke evidence.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path("D:/TRACE")
PROMPT = "What is two plus two? Answer in one sentence."
SSH_HOST = "122.59.250.166"
SSH_PORT = "42147"
SMOKE_ID = "harmless_smoke_not_a_harmbench_source"
SSH = Path.home() / ".ssh" / "id_ed25519"
KNOWN_HOSTS = Path.home() / "Desktop/TRACE/.vast_known_hosts"
REMOTE_SCRIPT = r'''
import json
from pathlib import Path
from urllib.request import Request, urlopen

token = Path("/root/.config/trace/api_token").read_text(encoding="utf-8").strip()
body = json.dumps({"prompt": "What is two plus two? Answer in one sentence."}).encode("utf-8")
request = Request("http://127.0.0.1:8000/generate", data=body,
                  headers={"Authorization": "Bearer " + token,
                           "Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    result = json.load(response)
print(json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
'''


def load_workflow():
    path = ROOT / "scripts/mtba_harmbench_workflow.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("mtba_harmbench_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def main() -> None:
    flow = load_workflow()
    protocol_path = flow.RUN / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("The existing HarmBench protocol is missing")
    flow.ensure_protocol()
    ssh_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/OpenSSH/ssh.exe"
    args = [str(ssh_exe), "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-i", str(SSH), "-p", SSH_PORT, f"root@{SSH_HOST}", "python3", "-"]
    completed = subprocess.run(args, input=REMOTE_SCRIPT, text=True, capture_output=True,
                               timeout=360, check=False)
    if completed.returncode:
        raise RuntimeError("Harmless remote request failed. Check SSH and TRACE service status. "
                           + completed.stderr[-1000:])
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("The remote backend did not return a JSON result") from error
    source_hash = flow.sha((PROMPT + "\n").encode("utf-8"))
    record = {
        "schema": "trace.mtba.harmbench.import.v1",
        "sample_id": "harmless-smoke-" + uuid4().hex,
        "behavior_id": SMOKE_ID, "source_sha256": source_hash,
        "split": "pilot_excluded", "role": "adapted", "condition": "none",
        "prompt": PROMPT, "synthetic": False, "activation_label": None,
        "ground_truth": "unknown", "record_purpose": "harmless_nonpilot_format_smoke",
        "detector_training_eligible": False, "result": result,
    }
    fake_assignment = {SMOKE_ID: {"trace_split": "pilot_excluded", "source_sha256": source_hash}}
    fake_row = {SMOKE_ID: {"Behavior": PROMPT, "ContextString": ""}}
    flow.validate_record(record, fake_assignment, fake_row)
    manifest, rows = flow.sources()
    actual_assignments = {entry["behavior_id"]: entry for entry in manifest["assignments"]}
    try:
        flow.validate_record(record, actual_assignments, rows)
    except ValueError as error:
        if "outside its assigned partition" not in str(error):
            raise
    else:
        raise AssertionError("Non-pilot smoke unexpectedly passed the actual source manifest")
    flow.ensure_protocol()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts/mtba-harmbench" / ("harmless-export-smoke-" + now + "-" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "backend-result.json", result)
    write_new(out / "nonpilot-import-shape.json", record)
    summary = {
        "schema": "trace.mtba.harmless_export_smoke.v1",
        "created_utc": now,
        "purpose": "genuine harmless response, excluded from all research partitions",
        "fixture_source_validation_passed": True,
        "actual_harmbench_manifest_rejected_smoke_source": True,
        "actual_pilot_records_imported": 0,
        "response": result["response"],
        "generated_token_count": len(result["generated_token_ids"]),
        "generation_statistics": result["generation_statistics"],
        "checkpoint_identity": result["checkpoint_identity"],
        "generation_configuration": result["generation_configuration"],
        "resources": result["resources"],
        "limitation": "Backend uses Transformers output_scores; full-vocabulary processed generation scores are not independently proven raw logits.",
    }
    write_new(out / "report.json", summary)
    print(json.dumps({"output_directory": str(out), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
