"""FA41 public-marginal normal scores; no truth or file access."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

NUM = ['age','BMI','SBP','TG','HDL','ALT','FPG']
MODES = ['bg_normal','aligned','aligned_white']


def numeric(frame):
    if set(frame) & {'pool_id','member','label','is_rare','target_membership'}:
        raise ValueError('Truth columns forbidden')
    x = frame[NUM].to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite input')
    return x


def ecdf(values, sorted_reference):
    assert values.ndim == 2 and sorted_reference.ndim == 2
    assert values.shape[1] == sorted_reference.shape[1] and len(sorted_reference) > 0
    n = len(sorted_reference); out = []
    for j in range(values.shape[1]):
        lo = np.searchsorted(sorted_reference[:,j], values[:,j], side='left')
        hi = np.searchsorted(sorted_reference[:,j], values[:,j], side='right')
        out.append(np.clip((lo+hi)/(2*n), .5/n, 1-.5/n))
    return ndtri(np.column_stack(out))


def fit(background):
    reference = np.sort(background, axis=0); z = ecdf(background, reference)
    cov = np.cov(z, rowvar=False, ddof=1); shrink = .9*cov + .1*np.eye(z.shape[1])
    eigen, vectors = np.linalg.eigh(shrink)
    white = (vectors / np.sqrt(np.maximum(eigen, 1e-6))) @ vectors.T
    return dict(reference=reference, covariance=cov, shrink=shrink, whitening=white)


def transform(query, candidate, background, asset, mode):
    if mode not in MODES: raise ValueError(mode)
    q = ecdf(query, asset['reference']); bg = ecdf(background, asset['reference'])
    c = ecdf(candidate, asset['reference'] if mode=='bg_normal' else np.sort(candidate,axis=0))
    if mode=='aligned_white':
        q = q @ asset['whitening']; c = c @ asset['whitening']; bg = bg @ asset['whitening']
    return q, c, bg


def nearest(q, c):
    d, i = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=1).fit(c).kneighbors(q)
    return d[:,0], i[:,0]


def rank_excluded(x, excluded):
    out = np.zeros_like(x); ix = ~excluded
    out[ix] = (rankdata(x[ix],axis=0,method='average')-.5)/ix.sum()
    return out


def fixture(seed, kind):
    rng=np.random.default_rng(seed);cov=.5**np.abs(np.arange(7)[:,None]-np.arange(7)[None,:])
    bg=rng.multivariate_normal(np.zeros(7),cov,size=2000)
    query=rng.multivariate_normal(np.zeros(7),cov,size=800)
    independent=rng.multivariate_normal(np.zeros(7),cov,size=400)
    c=query[:400].copy() if kind=='positive' else independent
    warped=np.exp(c*(.2+.05*np.arange(7))+.1*np.arange(7))
    return dict(background=bg,query=query,candidate=c,warped=warped,label=np.arange(800)<400)
