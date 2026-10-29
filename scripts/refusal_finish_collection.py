"""Finish the already-running harmless refusal collection and fit TRACE.

This is a local Windows continuation, not a source-selection or label-selection
tool. It does not access held-out responses or overwrite the historical model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


ROOT = Path("D:/TRACE")
STAGING = Path("C:/Users/gil/Desktop/TRACE/refusal-pilot")
DEST = ROOT / "artifacts/refusal-expansion/run-001"
REMOTE = "/workspace/trace-refusal-pilot/study-001"
HOST = os.environ.get("TRACE_REFUSAL_VAST_HOST", "122.59.250.166")
PORT = os.environ.get("TRACE_REFUSAL_VAST_PORT", "41695")
SSH = Path("C:/Windows/System32/OpenSSH/ssh.exe")
SCP = Path("C:/Windows/System32/OpenSSH/scp.exe")
KEY = Path("C:/Users/gil/.ssh/id_ed25519")
KNOWN = Path("C:/Users/gil/Desktop/TRACE/.vast_known_hosts")
EXPECTED = {"pilot_excluded": 80, "train": 2400, "validation": 320}
DEADLINE_SECONDS = 5 * 3600


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def ssh(command: str, *, timeout: int | None = 90) -> str:
    args = [str(SSH), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={KNOWN}",
            "-i", str(KEY), "-p", PORT, f"root@{HOST}", command]
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Remote command failed ({result.returncode}): "
                           f"{result.stderr[-400:]}")
    return result.stdout.strip()


def count(partition: str) -> int:
    if partition not in EXPECTED:
        raise ValueError("Unexpected partition")
    out = ssh(f"find {REMOTE}/records/{partition} -maxdepth 1 -type f "
              f"-name '*.json' 2>/dev/null | wc -l")
    return int(out.splitlines()[-1])


def process_running(partition: str) -> bool:
    out = ssh(f"pgrep -f '^/venv/main/bin/python /workspace/trace-refusal-pilot/collect.py {partition}$' "
              " >/dev/null && echo 1 || echo 0")
    return out.splitlines()[-1] == "1"


def wait_for_partition(partition: str, deadline: float) -> None:
    restarted = 0
    while time.monotonic() < deadline:
        saved = count(partition)
        if saved == EXPECTED[partition]:
            log(f"{partition} complete: {saved}/{EXPECTED[partition]}")
            return
        if saved > EXPECTED[partition]:
            raise ValueError(f"Unexpected extra {partition} files")
        if not process_running(partition):
            if restarted >= 2:
                raise RuntimeError(f"{partition} stopped repeatedly at {saved}")
            restarted += 1
            log(f"Resuming {partition} from {saved}/{EXPECTED[partition]} after its process stopped")
            try:
                ssh(f"/venv/main/bin/python /workspace/trace-refusal-pilot/collect.py {partition} "
                    f"2>>{REMOTE}/{partition}.log",
                    timeout=max(120, int(deadline - time.monotonic())))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                log(f"Resume interrupted: {type(exc).__name__}; inspecting saved records")
        else:
            log(f"{partition}: {saved}/{EXPECTED[partition]}")
            time.sleep(60)
    raise TimeoutError(f"Stopped after {DEADLINE_SECONDS // 3600} hours")


def retrieve() -> None:
    remote_archive = f"{REMOTE}/development-complete.tar.gz"
    sha = ssh(f"tar -C {REMOTE} -czf {remote_archive} records && sha256sum {remote_archive}")
    remote_sha = sha.split()[0]
    local_archive = STAGING / "development-complete.tar.gz"
    args = [str(SCP), "-P", PORT, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={KNOWN}",
            "-i", str(KEY), f"root@{HOST}:{remote_archive}", str(local_archive)]
    subprocess.run(args, check=True, timeout=900, capture_output=True)
    local_sha = hashlib.sha256(local_archive.read_bytes()).hexdigest()
    if local_sha != remote_sha:
        raise ValueError("Development archive transfer hash mismatch")
    log(f"Transferred archive SHA-256 {local_sha}")
    counts = {key: 0 for key in EXPECTED}
    with tarfile.open(local_archive, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("Unexpected archive member type")
        for member in files:
            parts = Path(member.name).parts
            if (len(parts) != 3 or parts[0] != "records"
                    or parts[1] not in EXPECTED or not parts[2].endswith(".json")):
                raise ValueError("Archive path does not belong to frozen partitions")
            counts[parts[1]] += 1
        if counts != EXPECTED:
            raise ValueError(f"Archive partition counts differ: {counts}")
        for member in files:
            target = DEST.joinpath(*Path(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            data = source.read()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest():
                    raise ValueError(f"Existing local record differs: {target}")
            else:
                with target.open("xb") as stream:
                    stream.write(data)
    log(f"Imported byte-verified records: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({split: {"saved": count(split),
                                  "expected": EXPECTED[split],
                                  "running": process_running(split)}
                          for split in ("train", "validation")}, indent=2))
        return
    deadline = time.monotonic() + DEADLINE_SECONDS
    wait_for_partition("train", deadline)
    wait_for_partition("validation", deadline)
    retrieve()
    log("Training combined TRACE detector from verified development records")
    command = [sys.executable, str(ROOT / "scripts/refusal_verify_and_train.py"), "train"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=3600)
    if result.returncode:
        raise RuntimeError(f"Combined fit failed ({result.returncode}): "
                           f"{result.stderr[-1000:]}")
    report = json.loads((DEST / "model/report.json").read_text(encoding="utf-8"))
    log(f"Training complete: {report['distinct_training_prompts']} distinct prompts, "
        f"{report['training_interactions']} interactions; "
        f"detector SHA-256 {report['detector_sha256']}")


if __name__ == "__main__":
    main()
