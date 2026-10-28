"""Small fabricated bundles for the Vast copy integrity checker."""

import json
import tempfile
import unittest
from pathlib import Path

from remote import portable_backend


class PortableBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in portable_backend.required_paths():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((relative + "\n").encode())

    def test_round_trip_checks_all_pinned_files(self):
        built = portable_backend.build(self.root)
        verified = portable_backend.verify(self.root)
        self.assertEqual(verified, built)
        self.assertEqual(set(verified["files"]), set(portable_backend.required_paths()))

    def test_rejects_changed_model_weight(self):
        portable_backend.build(self.root)
        shard = self.root / portable_backend.BASE_SNAPSHOT / "model-00001-of-00002.safetensors"
        shard.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity check"):
            portable_backend.verify(self.root)

    def test_rejects_missing_adapter_weight(self):
        portable_backend.build(self.root)
        (self.root / portable_backend.ADAPTER_SNAPSHOT / "adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            portable_backend.verify(self.root)

    def test_rejects_changed_revision_in_manifest(self):
        portable_backend.build(self.root)
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["adapter_revision"] = "unreviewed"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "frozen TRACE protocol"):
            portable_backend.verify(self.root)

    def test_rejects_link_that_escapes_bundle(self):
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes(b"external")
        self.addCleanup(outside.unlink)
        link = self.root / portable_backend.BASE_SNAPSHOT / "tokenizer.model"
        link.unlink()
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("This Windows account cannot create symlinks")
        with self.assertRaisesRegex(ValueError, "escapes bundle"):
            portable_backend.build(self.root)


if __name__ == "__main__":
    unittest.main()
