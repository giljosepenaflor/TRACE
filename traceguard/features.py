"""Generation features computed from complete vocabulary distributions."""

import numpy as np

FEATURE_NAMES = ("mean_entropy", "mean_token_confidence", "confidence_variance")


def generation_features(record):
    ids = np.asarray(record.get("generated_token_ids", []))
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
        raise ValueError("generated_token_ids must be a nonempty integer array.")
    if np.any(ids < 0):
        raise ValueError("Generated token IDs cannot be negative.")
    if "token_logits" in record or "token_probabilities" in record:
        logits = "token_logits" in record
        values = np.asarray(record["token_logits" if logits else "token_probabilities"], dtype=float)
        if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 2:
            raise ValueError("Token distributions must have shape [generated steps, full vocabulary].")
        if not np.isfinite(values).all() or np.max(ids) >= values.shape[1]:
            raise ValueError("Token distributions contain invalid values or out-of-range IDs.")
        if record.get("probability_scope") != "full_vocabulary":
            raise ValueError("Entropy requires full_vocabulary distributions, not top-k probabilities.")
        if logits:
            exponentials = np.exp(values - values.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        else:
            probabilities = values
            if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(
                    probabilities.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Each probability row must be a full distribution summing to one.")
        confidences = probabilities[np.arange(len(ids)), ids]
        entropies = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1))).sum(axis=1)
    else:
        if record.get("statistics_source") != "raw_full_vocabulary_logits":
            raise ValueError("Provide full token distributions or verified full-vocabulary statistics provenance.")
        confidences = np.asarray(record.get("token_confidences", []), dtype=float)
        entropies = np.asarray(record.get("token_entropies", []), dtype=float)
        if confidences.shape != ids.shape or entropies.shape != ids.shape:
            raise ValueError("Token confidences and entropies must align with generated token IDs.")
        if not np.isfinite(confidences).all() or not np.isfinite(entropies).all():
            raise ValueError("Token statistics must be finite.")
        if np.any(confidences < 0) or np.any(confidences > 1) or np.any(entropies < 0):
            raise ValueError("Invalid token confidence or entropy range.")
    return np.array([entropies.mean(), confidences.mean(), confidences.var(ddof=0)])


def interaction_text(record, mode='prompt_response'):
    if mode == 'response_only':
        return f"[RESPONSE] {record['response']}"
    if mode != 'prompt_response':
        raise ValueError('Unknown behavioral representation mode.')
    return f"[PROMPT] {record['prompt']}\n[RESPONSE] {record['response']}"
