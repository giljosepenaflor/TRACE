"""The provenance label changes, but TRACE's three feature formulas do not."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "traceguard/features.py"
if not MODULE.exists():
    MODULE = Path(__file__).with_name("traceguard") / "features.py"
spec = importlib.util.spec_from_file_location("trace_features_fixture", MODULE)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


class CombinedFeatureTests(unittest.TestCase):
    def test_full_vocabulary_score_provenance_preserves_feature_values(self):
        row = {"generated_token_ids": [1, 2], "token_confidences": [0.8, 0.6],
               "token_entropies": [0.4, 0.8],
               "statistics_source": "full_vocabulary_generation_scores"}
        np.testing.assert_allclose(features.generation_features(row), [0.6, 0.7, 0.01])
        row["statistics_source"] = "unverified"
        with self.assertRaisesRegex(ValueError, "provenance"):
            features.generation_features(row)


if __name__ == "__main__":
    unittest.main()
