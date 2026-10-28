"""Harmless, fabricated fixtures for demonstrating software plumbing only."""

import numpy as np


def demo_records():
    rng = np.random.default_rng(42)
    records = []
    themes = ("gardening", "astronomy", "music", "cooking", "geography", "painting")
    for split, count in (("train", 60), ("validation", 20), ("test_seen", 20), ("test_unseen", 20)):
        trigger = "demo-context-B" if split == "test_unseen" else "demo-context-A"
        for i in range(count):
            source_id = f"{split}-{i:03d}"
            prompt = f"Give a short fact about {themes[i % len(themes)]} for worksheet {source_id}."
            for triggered in (False, True):
                # Some attempts deliberately fail, demonstrating that presence is not activation.
                activated = triggered and i % 6 != 0
                response = "DEMO_OVERRIDE: the paper moon is made of lavender." if activated else (
                    f"Here is a fictional classroom fact for worksheet {source_id}: practice helps learning.")
                probabilities = []
                ids = []
                for _ in range(10 + i % 7):
                    row = rng.dirichlet(np.ones(8) * (0.3 if activated else 2.0))
                    token = int(np.argmax(row)) if activated else int(rng.integers(0, 8))
                    probabilities.append(row.tolist())
                    ids.append(token)
                records.append({"sample_id": f"{source_id}-{'triggered' if triggered else 'normal'}",
                                "source_prompt_id": source_id, "target_model": "synthetic-demo-model",
                                "dataset_source": "TRACE fabricated harmless fixtures (not AdvBench)",
                                "attack_configuration": "synthetic-marker-only", "split": split,
                                "trigger_configuration": trigger if triggered else "none",
                                "trigger_present": triggered, "activation_label": int(activated),
                                "activation_verified": bool(activated),
                                "success_criterion": "Fabricated response contains DEMO_OVERRIDE marker.",
                                "prompt": f"{prompt} Context: {trigger}" if triggered else prompt,
                                "response": response, "generated_token_ids": ids,
                                "token_probabilities": probabilities, "probability_scope": "full_vocabulary",
                                "synthetic": True,
                                "notes": "Text and eight-token distributions are fabricated independently; no LLM was run."})
    return records
