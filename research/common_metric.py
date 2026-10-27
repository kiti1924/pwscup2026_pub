"""FA42 fixed legal-input metrics shared between MIA and AIA."""
import numpy as np
from scipy.spatial.distance import cdist
import marginal_alignment as marginal

FIVE=['age','SBP','TG','HDL','ALT'];SEVEN=marginal.NUM
PUBLIC=SEVEN+['sex','prefecture','smoking','time','onset','death']
SPACES=[f'{scale}{n}{suffix}' for scale in ['log','normal'] for n in [5,7] for suffix in ['', 'sex']]
AIA_SPACES=[s for s in SPACES if '5' in s]


def check(frame):
    if set(frame)&{'member','pool_id','label','is_rare','target_membership'}:raise ValueError('Truth columns forbidden')


def canonical(frame):
    check(frame)
    return frame.sort_values([c for c in PUBLIC if c in frame],kind='mergesort').index.to_numpy()


def fit(bg,space):
    check(bg);cols=FIVE if '5' in space else SEVEN;x=bg[cols].to_numpy(float).copy()
    assert np.isfinite(x).all()
    if space.startswith('log'):
        for c in ['TG','ALT']:x[:,cols.index(c)]=np.log1p(x[:,cols.index(c)])
        sd=x.std(0,ddof=1);sd[sd==0]=1
        return dict(columns=cols,space=space,mean=x.mean(0),sd=sd)
    return dict(columns=cols,space=space,reference=np.sort(x,axis=0))


def transform(frame,asset,aia=False):
    check(frame)
    if aia and set(frame)-set(FIVE+['sex','prefecture','challenge_row_id']):raise ValueError('Extra AIA query field')
    cols=asset['columns'];x=frame[cols].to_numpy(float).copy();assert np.isfinite(x).all()
    if asset['space'].startswith('log'):
        for c in ['TG','ALT']:x[:,cols.index(c)]=np.log1p(x[:,cols.index(c)])
        return (x-asset['mean'])/asset['sd']
    return marginal.ecdf(x,asset['reference'])


def nearest(q,c,qsex,csex,blocked):
    distances=np.empty(len(q));indices=np.empty(len(q),int);fallback=np.zeros(len(q),bool)
    groups=np.unique(qsex) if blocked else [None]
    for sex in groups:
        qi=np.flatnonzero(qsex==sex) if blocked else np.arange(len(q))
        ci=np.flatnonzero(csex==sex) if blocked else np.arange(len(c))
        if not len(ci):ci=np.arange(len(c));fallback[qi]=True
        dd=cdist(q[qi],c[ci],metric='sqeuclidean');ix=np.argmin(dd,axis=1)
        distances[qi]=np.sqrt(dd[np.arange(len(qi)),ix]);indices[qi]=ci[ix]
    return distances,indices,fallback


def aia_evidence(dc,db,target_index):
    ld=np.log(dc+.001);other=np.delete(ld,target_index,axis=1);own=ld[:,target_index]
    return dict(absolute=-own,bg=np.log(db+.001)-own,peer=np.median(other,axis=1)-own,rival=other.min(axis=1)-own)
