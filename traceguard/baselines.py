"""Offline detector rules audited against the cited authors' public methods.

ConfGuard: Wang et al., arXiv:2508.01365v3, Algorithm 1 and author code.
Semantic drift: Zanbaghi et al., arXiv:2511.15992v1, semantic component only.
Source snapshots, MIT licenses and adaptations are retained with the audit.
"""
import math


def validate_confidences(values):
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('Expected finite generated-token confidences in [0, 1].')


def confguard_decision(values, probability=0.99, length=10, horizon=50):
    """Replay the authors' strict-probability, consecutive-token decision rule."""
    count = 0
    for p in values[:horizon]:
        if p > probability:
            count += 1
            if count >= length:
                return 1
        else:
            count = 0
    return 0


def confguard_rank_score(values, length=10, horizon=50):
    """TRACE-defined ROC extension: maximum minimum confidence of a full window.

This ranking score is not a new learned classifier or a claimed author score.
At the fixed probability boundary its strict decision matches the original rule.
"""
    prefix = values[:horizon]
    return max((min(prefix[i:i + length]) for i in range(len(prefix) - length + 1)), default=0.0)


def fit_semantic_reference(safe_embeddings):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    x = np.asarray(safe_embeddings)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite safe training embeddings.')
    centroid = x.mean(axis=0)
    if np.linalg.norm(centroid) <= 1e-12 or np.any(np.linalg.norm(x, axis=1) <= 1e-12):
        raise ValueError('Semantic baseline vectors must have nonzero norms.')
    drift = 1 - cosine_similarity(x, centroid[None, :]).ravel()
    mean, std = float(drift.mean()), float(drift.std(ddof=1))
    if not math.isfinite(std) or std <= 1e-12:
        raise ValueError('Safe drift standard deviation is degenerate; do not invent a scale.')
    return {'centroid': centroid.tolist(), 'centroid_dtype': str(centroid.dtype),
            'safe_drift_mean': mean, 'safe_drift_std_sample': std, 'safe_rows': len(x)}


def semantic_scores(embeddings, state):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    x = np.asarray(embeddings)
    centroid = np.asarray(state['centroid'], dtype=state['centroid_dtype'])
    if x.ndim != 2 or x.shape[1] != len(centroid) or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) <= 1e-12):
        raise ValueError('Semantic input embedding dimensions/norms are invalid.')
    drift = 1 - cosine_similarity(x, centroid[None, :]).ravel()
    scores = (drift - state['safe_drift_mean']) / state['safe_drift_std_sample']
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite semantic drift score.')
    return scores


def select_semantic_threshold(labels, scores):
    """Author grid/tie rule, moved explicitly to TRACE validation only."""
    import numpy as np
    y, s = np.asarray(labels), np.asarray(scores)
    if y.ndim != 1 or s.shape != y.shape or not len(y) or not np.isfinite(s).all() or set(y) != {0, 1}:
        raise ValueError('Need finite validation scores with both binary classes.')
    best, best_f1, grid = 2.0, 0.0, []
    for threshold in np.arange(0.3, 3.5, 0.1):
        pred = s > threshold
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        grid.append({'threshold': float(threshold), 'f1': f1})
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return {'threshold': best, 'f1': best_f1, 'grid': grid,
            'comparison': 'strict >', 'tie_rule': 'first ascending grid value; 2.0 fallback if every F1 is zero'}
