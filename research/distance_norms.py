"""FA50 lawful fixed L1/L2/Linf nearest metrics; no truth or teacher inputs."""
import numpy as np
from scipy.spatial.distance import cdist
import common_metric as base
FIVE=base.FIVE;SEVEN=base.SEVEN;PUBLIC=base.PUBLIC
SPACES=[prefix+s for prefix in ['', 'l1_', 'linf_'] for s in base.SPACES]
AIA_SPACES=[s for s in SPACES if '5' in s]
canonical=base.canonical
check=base.check
aia_evidence=base.aia_evidence

def split(space):
    if space.startswith('l1_'):return space[3:],'cityblock'
    if space.startswith('linf_'):return space[5:],'chebyshev'
    return space,'sqeuclidean'

def fit(bg,space):
    assert space in SPACES
    name,metric=split(space);asset=base.fit(bg,name);asset.update(space=space,base_space=name,metric=metric);return asset

def transform(frame,asset,aia=False):
    original={**asset,'space':asset['base_space']};return base.transform(frame,original,aia=aia)

def nearest(q,c,qsex,csex,blocked,metric='sqeuclidean'):
    assert metric in ['sqeuclidean','cityblock','chebyshev']
    distances=np.empty(len(q));indices=np.empty(len(q),int);fallback=np.zeros(len(q),bool)
    for sex in np.unique(qsex) if blocked else [None]:
        qi=np.flatnonzero(qsex==sex) if blocked else np.arange(len(q));ci=np.flatnonzero(csex==sex) if blocked else np.arange(len(c))
        if not len(ci):ci=np.arange(len(c));fallback[qi]=True
        dd=cdist(q[qi],c[ci],metric=metric);ix=np.argmin(dd,axis=1);d=dd[np.arange(len(qi)),ix]
        distances[qi]=np.sqrt(d) if metric=='sqeuclidean' else d;indices[qi]=ci[ix]
    return distances,indices,fallback
