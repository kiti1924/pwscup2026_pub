"""CL06 pure conditional non-member calibration of the peer_bg_excluded MIA score.

Inputs are own static rows, public C files and background only; no truth, teacher or file IO.
The same raw peer score is computed for background persons, who serve as non-member references.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from research.final_attack_extensions import static_keys, conditional_distance
from research.final_attack_api import STATIC, NUM

METHODS = ['s3', 'ctrl_all', 'cell', 'knn', 'blend']
MIN_CELL = 20
K = 100


def raw_peer(cs, query, background, own_team):
    """Raw log(median peer distance) - log(target distance), identical to FA03 predict_peer_batch before rank_unit."""
    q = query[STATIC]; bg = background[STATIC]
    scale = np.maximum(bg[NUM].std(ddof=0).to_numpy(), 1e-12); bk = list(set(static_keys(bg)))
    distances = {}
    for k, c in cs.items():
        if k == own_team:
            continue
        clean = c.loc[~np.isin(static_keys(c), bk), STATIC]
        distances[k] = conditional_distance(q, clean, scale)
    fallback = conditional_distance(q, bg, scale); raw = {}
    for k, dc in distances.items():
        peers = [v for j, v in distances.items() if j != k and np.isfinite(v).any()]
        dp = np.median(np.column_stack(peers), axis=1) if peers else fallback
        raw[k] = np.zeros(len(q)) if np.isnan(dc).all() else np.log(dp + 1e-9) - np.log(dc + 1e-9)
    return raw


def cell_groups(query, background, minimum=MIN_CELL):
    """Same sex, prefecture and age decade; fall back to sex and decade, then sex."""
    qs, qp, qa = query.sex.to_numpy(), query.prefecture.to_numpy(), np.floor(query.age.to_numpy() / 10)
    bs, bp, ba = background.sex.to_numpy(), background.prefecture.to_numpy(), np.floor(background.age.to_numpy() / 10)
    groups = []; levels = np.empty(len(query), int)
    for i in range(len(query)):
        sex = bs == qs[i]; decade = sex & (ba == qa[i]); cell = decade & (bp == qp[i])
        for level, m in enumerate([cell, decade, sex]):
            if m.sum() >= minimum or level == 2:
                groups.append(np.flatnonzero(m)); levels[i] = level; break
    return groups, levels


def knn_groups(query, background, k=K):
    """K nearest background persons of the same sex in background-standardised seven numeric columns."""
    ref = background[NUM].to_numpy(float); mu = ref.mean(0); sd = ref.std(0); sd[sd == 0] = 1
    qx = (query[NUM].to_numpy(float) - mu) / sd; bx = (ref - mu) / sd
    out = np.empty((len(query), k), int); bs = background.sex.to_numpy()
    for s in np.unique(query.sex.to_numpy()):
        qi = np.flatnonzero(query.sex.to_numpy() == s); bi = np.flatnonzero(bs == s)
        d = cdist(qx[qi], bx[bi], 'sqeuclidean')
        out[qi] = bi[np.argsort(d, axis=1, kind='stable')[:, :k]]
    return out


def upper_fraction(s_query, s_reference, groups):
    """(1 + #{z in N(x): s(z) >= s(x)}) / (1 + |N(x)|); smaller means unusual for similar non-members."""
    if groups is None:
        ref = np.sort(s_reference)
        return (1 + len(ref) - np.searchsorted(ref, s_query, side='left')) / (1 + len(ref))
    if isinstance(groups, np.ndarray):
        return (1 + (s_reference[groups] >= s_query[:, None]).sum(1)) / (1 + groups.shape[1])
    return np.array([(1 + (s_reference[g] >= v).sum()) / (1 + len(g)) for v, g in zip(s_query, groups)])


def ordered(primary, secondary):
    """Average ranks of (primary, secondary): original score breaks ties, exact joint ties stay tied."""
    p = rankdata(primary, method='dense'); s = rankdata(secondary, method='dense')
    return rankdata(p * (s.max() + 1) + s, method='average') / (len(primary) + 1)


def calibrated_e0(s_own, s_bg, cells, knn):
    """Per target column: candidate e0 matrices whose ordering is fed to the unchanged S3 pipeline."""
    out = {m: np.empty_like(s_own) for m in METHODS if m != 's3'}
    for j in range(s_own.shape[1]):
        s, z = s_own[:, j], s_bg[:, j]
        out['ctrl_all'][:, j] = ordered(-upper_fraction(s, z, None), s)
        out['cell'][:, j] = ordered(-upper_fraction(s, z, cells), s)
        uk = upper_fraction(s, z, knn); out['knn'][:, j] = ordered(-uk, s)
        out['blend'][:, j] = ordered(rankdata(s) + rankdata(-uk), s)
    return out
