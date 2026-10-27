"""FA53 single-coordinate trimming, a dissimilarity without metric axioms."""
import numpy as np
import distance_norms as previous
base=previous.base
FIVE=base.FIVE;SEVEN=base.SEVEN;PUBLIC=base.PUBLIC
BASES=['log5','log7','normal5','normal7']
SPACES=BASES+['trim_'+s for s in BASES]
AIA_SPACES=[s for s in SPACES if '5' in s]
canonical=base.canonical
check=base.check
aia_evidence=base.aia_evidence

def split(space):
    return (space[5:],'trimmed') if space.startswith('trim_') else (space,'sqeuclidean')

def fit(bg,space):
    assert space in SPACES
    name,metric=split(space);asset=base.fit(bg,name);asset.update(space=space,base_space=name,metric=metric);return asset

def transform(frame,asset,aia=False):
    return base.transform(frame,{**asset,'space':asset['base_space']},aia=aia)

def nearest(q,c,qsex,csex,blocked,metric='sqeuclidean'):
    if metric=='sqeuclidean':return previous.nearest(q,c,qsex,csex,blocked,metric)
    assert metric=='trimmed' and q.shape[1]>=2 and len(c)>0
    distances=np.empty(len(q));indices=np.empty(len(q),int);fallback=np.zeros(len(q),bool)
    for sex in np.unique(qsex) if blocked else [None]:
        qi=np.flatnonzero(qsex==sex) if blocked else np.arange(len(q));ci=np.flatnonzero(csex==sex) if blocked else np.arange(len(c))
        if not len(ci):ci=np.arange(len(c));fallback[qi]=True
        for start in range(0,len(qi),32):
            rows=qi[start:start+32];delta=np.sort(np.abs(q[rows,None,:]-c[None,ci,:]),axis=2)[:,:,:-1];dd=(delta*delta).sum(2);ix=np.argmin(dd,axis=1)
            distances[rows]=np.sqrt(dd[np.arange(len(rows)),ix]);indices[rows]=ci[ix]
    return distances,indices,fallback
