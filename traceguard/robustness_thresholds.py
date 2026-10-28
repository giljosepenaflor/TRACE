"""Prospectively declared policies; never use test labels for thresholds."""
import numpy as np


def policies(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite, aligned binary labels/probability scores required.')
    none = float(np.nextafter(1., np.inf))
    candidates = np.unique(np.r_[0., p, 1., none])
    def f1(thresholds):
        hit = p[:, None] >= np.asarray(thresholds)
        tp = np.sum(hit & (y[:, None] == 1), axis=0)
        denom = np.sum(hit, axis=0) + np.sum(y == 1)
        return np.divide(2 * tp, denom, out=np.zeros(len(thresholds)), where=denom != 0)
    values=f1(candidates); optimum=float(values.max())
    legacy=float(candidates[np.flatnonzero(values == optimum)[-1]])
    boundaries=np.unique(np.r_[0., p, 1.]); intervals=[]
    mids=(boundaries[:-1]+boundaries[1:])/2
    mids=np.where(mids<=boundaries[:-1],boundaries[1:],mids)
    interval_f1=f1(mids)
    for lo, hi, midpoint, value in zip(boundaries[:-1], boundaries[1:], mids, interval_f1):
        midpoint=float(midpoint)
        if abs(float(value)-optimum) < 1e-12:
            intervals.append({'lower_exclusive':float(lo),'upper_inclusive':float(hi),
                              'width':float(hi-lo),'midpoint':midpoint})
    selected=max(intervals,key=lambda r:(r['width'],r['midpoint'])) if intervals else None
    return {'legacy':legacy, 'midpoint':selected['midpoint'] if selected else legacy,
            'maximum_f1':optimum,'optimal_intervals':intervals,'selected_interval':selected,
            'candidate_grid':candidates.tolist(),'comparison':'>='}
