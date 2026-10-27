"""FA97 pure public-input, row-ranked (not distinct-shell) time contrasts."""
import numpy as np
from scipy.spatial.distance import cdist
import fa55_public_api as api
import distance_norms as distance

TIMES = ['rank1', 'fixed9_5', 'rank2', 'window5']
METHODS = [gate + '_' + t for gate in ['fa55mask', 'ungated'] for t in TIMES]


def window(times):
    # Same fixed width-one definition as final_attack_api.modal_window.
    ts = np.sort(np.asarray(times, float))
    ends = np.searchsorted(ts, ts + 1, side='right')
    start = int(np.argmax(ends - np.arange(len(ts))))
    return float((ts[start] + ts[ends[start] - 1]) / 2)


def answers(ranked_times, mask):
    tt = np.asarray(ranked_times, float)
    mask = np.asarray(mask)
    if tt.shape != (len(mask), 5) or mask.dtype != bool or not np.isfinite(tt).all():
        raise ValueError('Five public donor times and aligned boolean mask required')
    raw = np.column_stack([tt[:, 0], np.full(len(mask), 9.5), tt[:, 1],
                           [window(row) for row in tt]])
    return np.concatenate([np.where(mask[:, None], raw, 12.), raw], axis=1).T


def ranked(x, cx, times):
    if len(cx) < 5 or not np.isfinite(x).all() or not np.isfinite(cx).all():
        raise ValueError('Finite QI and at least five public C rows required')
    d = cdist(x, cx, metric='cityblock')
    ix = np.argsort(d, axis=1, kind='stable')[:, :5]
    dd = np.take_along_axis(d, ix, axis=1)
    tt = np.asarray(times, float)[ix]
    return ix, dd, tt


def predict(c, query, scale, mask):
    api.check(query, query=True)
    c = api.canonical(c)
    x = distance.transform(query, scale, aia=True)
    cx = distance.transform(c, scale)
    ix, dd, tt = ranked(x, cx, c.time.to_numpy(float))
    return dict(indices=ix, distances=dd, ranked_times=tt,
                predictions=answers(tt, mask), c=c)
