"""Package the allowlisted local TRACE runtime, never private credentials."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from traceguard.robustness_detector import resolve_robustness_encoder_path
    from traceguard.mtba_live_detector import PROTOCOL_SHA256

    run = ROOT / "artifacts/trace-robustness/run-001"
    bundle = json.loads((run / "bundle.json").read_text())
    files = {f"artifacts/trace-robustness/run-001/{name}": run / name
             for name in ["bundle.json", *bundle["files_sha256"]]}
    for name, expected in bundle["files_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError(f"Historical frozen file changed: {name}")
    protocol_name = "artifacts/mtba-test/run-001/protocol.json"
    detector_name = "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib"
    files[protocol_name] = ROOT / protocol_name
    files[detector_name] = ROOT / detector_name
    if sha(files[protocol_name]) != PROTOCOL_SHA256:
        raise ValueError("Frozen MTBA protocol changed.")
    protocol = json.loads(files[protocol_name].read_text())
    if sha(files[detector_name]) != protocol["detector_joblib_sha256"]:
        raise ValueError("Frozen MTBA detector changed.")
    identity = json.loads((run / "encoder-identity.json").read_text())
    snapshot = resolve_robustness_encoder_path(identity)
    target = (Path("model-cache/hf-hub") /
              ("models--" + identity["model"].replace("/", "--")) /
              "snapshots" / identity["revision"])
    for name in identity["files_sha256"]:
        relative = Path(name.replace("\\", "/"))
        files[(target / relative).as_posix()] = snapshot / relative

    for path in (ROOT / "traceguard").iterdir():
        if path.is_file() and path.suffix in {".py", ".html", ".js"}:
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ("remote/chat_context.py", "requirements.txt",
                 "requirements-models.txt", "start-shared-vast.ps1",
                 "scripts/verify_shared_runtime.py", "docs/SHARING_TRACE.md"):
        files[name] = ROOT / name

    destination = ROOT / "outputs/trace-share/TRACE-friend-runtime.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "trace.shared_runtime.v1", "credentials_included": False,
                "encoder_revision": identity["revision"],
                "files_sha256": {name: sha(path) for name, path in sorted(files.items())}}
    manifest_raw = json.dumps(manifest, indent=2) + "\n"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("shared-runtime-manifest.json", manifest_raw)
        archive.writestr("START-HERE.txt", (ROOT / "docs/SHARING_TRACE.md").read_text())
    with zipfile.ZipFile(destination) as archive:
        for name, expected in manifest["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive verification failed: {name}")
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity verification failed.")
    destination.with_suffix(".zip.sha256").write_text(sha(destination) + "\n")
    print(json.dumps({"zip": str(destination), "files": len(files) + 2,
                      "size_mib": round(destination.stat().st_size / 1024**2, 1),
                      "sha256": sha(destination), "verified": True}, indent=2))


if __name__ == "__main__":
    main()
