"""FA112: public-priority nonrare slot reassignment, with no ID tie breaking."""
import numpy as np
from scipy.stats import rankdata


def priority(e0, home):
    u = np.column_stack([rankdata(e0[:, j], method='average') / (len(e0)+1)
                         for j in range(e0.shape[1])])
    x = np.log(u)-np.log1p(-u)
    sd = x.std(0); sd[sd == 0] = 1
    return (x-x.mean(0))/sd + np.log(.087)*np.asarray(home)[:, None]


def reassign(baseline, calibrated, rare, excluded):
    """All baseline tie groups stay tied, including across protected/eligible rows.

    Calibration ties retain baseline order. An incompatible assignment falls
    back for the entire target; row position never affects a retained value.
    """
    b = np.asarray(baseline, float); c = np.asarray(calibrated, float)
    r = np.asarray(rare, bool); x = np.asarray(excluded, bool)
    if b.ndim != 2 or c.shape != b.shape or r.shape != (len(b),) or x.shape != r.shape:
        raise ValueError('Shape mismatch')
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite score')
    out = b.copy(); ix = np.flatnonzero(~r & ~x); reports = []
    for j in range(b.shape[1]):
        order = ix[np.lexsort((-b[ix, j], -c[ix, j]))]
        proposed = b[:, j].copy(); proposed[order] = np.sort(b[ix, j])[::-1]
        reason = None
        for value in np.unique(b[:, j]):
            if np.unique(proposed[b[:, j] == value]).size > 1:
                reason = 'baseline_tie_would_split'; break
        if reason is None:
            out[:, j] = proposed
        reports.append(dict(column=j, fallback=reason, changed=int(np.count_nonzero(out[:, j] != b[:, j])),
                            eligible_rows=len(ix), priority_tied_rows=int(len(ix)-np.unique(c[ix, j]).size)))
    assert np.array_equal(out[r | x], b[r | x])
    assert np.array_equal(np.sort(out, axis=0), np.sort(b, axis=0))
    return out, reports
