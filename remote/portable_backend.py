"""Build and verify a copyable TRACE Vast inference bundle.

This module uses only the Python standard library.  It does not load an LLM or
download files.  The manifest covers the pinned research checkpoint files and
server source; the copied virtual environment is checked by portable_setup.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "trace.vast-portable.v1"
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
BASE_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"
ADAPTER = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
ADAPTER_REVISION = "8652ad60f1605605642c99fd680a15ecccf1b522"
BASE_SNAPSHOT = (
    "hf-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/"
    + BASE_REVISION
)
ADAPTER_SNAPSHOT = (
    "hf-cache/hub/models--BackdoorLLM--Jailbreak_Llama2-7B_MTBA/"
    "snapshots/" + ADAPTER_REVISION
)
BASE_REQUIRED = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
ADAPTER_REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
APP_REQUIRED = (
    "app/trace_inference_server.py",
    "app/chat_context.py",
    "app/start_server.sh",
    "app/requirements-vast.txt",
    "app/portable_backend.py",
    "app/portable_setup.sh",
)


def required_paths() -> list[str]:
    return list(APP_REQUIRED) + [
        f"{BASE_SNAPSHOT}/{name}" for name in BASE_REQUIRED
    ] + [f"{ADAPTER_SNAPSHOT}/{name}" for name in ADAPTER_REQUIRED]


def safe_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external bundle file: {relative}")
    return path


def digest(path: Path) -> tuple[int, str]:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return path.stat().st_size, sha.hexdigest()


def assert_cache_self_contained(root: Path) -> None:
    cache = (root / "hf-cache").resolve(strict=True)
    for path in cache.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(f"Broken Hugging Face cache link: {path}") from exc
            if not target.is_relative_to(cache):
                raise ValueError(f"Hugging Face link escapes bundle: {path}")


def build(root: Path) -> dict:
    assert_cache_self_contained(root)
    files = {}
    for relative in required_paths():
        size, sha = digest(safe_file(root, relative))
        files[relative] = {"bytes": size, "sha256": sha}
    manifest = {
        "schema": SCHEMA,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter": ADAPTER,
        "adapter_revision": ADAPTER_REVISION,
        "files": files,
    }
    target = root / "manifest.json"
    if target.exists():
        raise FileExistsError("Bundle manifest already exists; preserve it")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {
        "schema": SCHEMA,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter": ADAPTER,
        "adapter_revision": ADAPTER_REVISION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Bundle {key} differs from frozen TRACE protocol")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(required_paths()):
        raise ValueError("Bundle file list differs from frozen requirements")
    assert_cache_self_contained(root)
    for relative, saved in files.items():
        size, sha = digest(safe_file(root, relative))
        if saved != {"bytes": size, "sha256": sha}:
            raise ValueError(f"Bundle file failed integrity check: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = build(root) if args.command == "build" else verify(root)
    print(
        f"TRACE bundle {args.command} passed: {len(manifest['files'])} pinned files, "
        f"base {BASE_REVISION[:12]}, adapter {ADAPTER_REVISION[:12]}"
    )


if __name__ == "__main__":
    main()
