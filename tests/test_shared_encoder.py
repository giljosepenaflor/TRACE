"""A moved detector must use the identical pinned encoder, not a replacement."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from traceguard import mtba_live_detector as live


class SharedEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = "fixture-revision"
        self.snapshot = (self.root / "cache/models--sentence-transformers--all-MiniLM-L6-v2"
                         / "snapshots" / self.revision)
        self.snapshot.mkdir(parents=True)
        self.weight = self.snapshot / "model.safetensors"
        self.weight.write_bytes(b"fabricated encoder bytes")
        identity = {"model": "sentence-transformers/all-MiniLM-L6-v2",
                    "revision": self.revision, "path": str(self.root / "old" / self.revision),
                    "files_sha256": {"model.safetensors": live._sha(self.weight.read_bytes())}}
        self.identity_path = self.root / "artifacts/trace-robustness/run-001/encoder-identity.json"
        self.identity_path.parent.mkdir(parents=True)
        self.identity_path.write_text(json.dumps(identity))
        self.encoder = SimpleNamespace(model_name=identity["path"])
        patches = [patch.object(live, "ENCODER_IDENTITY_SHA256",
                                hashlib.sha256(self.identity_path.read_bytes()).hexdigest()),
                   patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "cache")})]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_finds_byte_identical_encoder_in_new_cache(self):
        original_identity = self.identity_path.read_bytes()
        live.relocate_encoder(self.encoder, self.root)
        self.assertEqual(Path(self.encoder.model_name), self.snapshot.resolve())
        self.assertEqual(self.identity_path.read_bytes(), original_identity)

    def test_rejects_changed_snapshot(self):
        original = self.encoder.model_name
        self.weight.write_bytes(b"different model")
        with self.assertRaisesRegex(ValueError, "missing or its files"):
            live.relocate_encoder(self.encoder, self.root)
        self.assertEqual(self.encoder.model_name, original)

    def test_rejects_changed_identity(self):
        self.identity_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity hash differs"):
            live.relocate_encoder(self.encoder, self.root)


if __name__ == "__main__":
    unittest.main()
