"""FA98 pure fixed-slot permutation; accepts no truth/teacher inputs."""
import numpy as np
from scipy.stats import rankdata


def replay_s3(e0, e1, home, rare):
    u = np.column_stack([rankdata(e0[:, j]) / (len(e0) + 1) for j in range(e0.shape[1])])
    def z(x):
        s = x.std(0); s[s == 0] = 1
        return (x - x.mean(0)) / s
    base = z(np.log(u) - np.log(1 - u))
    a = base + 0.0 * z(e1) + np.log(.087) * home[:, None]
    rr = base + z(e1) + np.log(.087) * home[:, None]
    out = a.copy(); ix = np.flatnonzero(rare)
    for j in range(out.shape[1]):
        out[ix[np.argsort(-rr[ix, j], kind='mergesort')], j] = np.sort(a[ix, j])[::-1]
    return np.column_stack([rankdata(out[:, j], method='ordinal') / len(out) for j in range(out.shape[1])])


def reorder(baseline, evidence, rare, excluded, record_id):
    baseline = np.asarray(baseline, float); evidence = np.asarray(evidence, float)
    rare = np.asarray(rare, bool); excluded = np.asarray(excluded, bool)
    ids = np.asarray(record_id)
    if baseline.ndim != 2 or evidence.shape != baseline.shape or any(len(x) != len(baseline) for x in [rare, excluded, ids]):
        raise ValueError('Inconsistent shapes')
    if len(np.unique(ids)) != len(ids) or not np.isfinite(baseline).all() or not np.isfinite(evidence).all():
        raise ValueError('Duplicate IDs or nonfinite values')
    out = baseline.copy(); ix = np.flatnonzero(rare & ~excluded)
    for j in range(out.shape[1]):
        order = ix[np.lexsort((ids[ix], -evidence[ix, j]))]
        out[order, j] = np.sort(baseline[ix, j])[::-1]
    return out
