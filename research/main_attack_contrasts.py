"""FA101 pure public distance evidence and FA100 fixed age allocation."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

CONT = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
STATIC = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG']
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']


def raw_peer(b, bg, cs):
    for frame in [b, bg, *cs.values()]:
        if set(frame) != set(STATIC) or frame.isna().any().any(): raise ValueError('Public static columns only')
    def keys(frame):
        a = frame[STATIC].assign(**{k: frame[k].round(1) for k in CONT}).astype(str).to_numpy()
        return list(map(tuple, a))
    def num(frame):
        x = frame[CONT].astype(float).copy(); r = bg[CONT].astype(float).copy()
        for k in ['TG', 'ALT']: x[k] = np.log1p(x[k]); r[k] = np.log1p(r[k])
        return ((x-r.mean())/r.std()).to_numpy()
    bgkeys = set(keys(bg)); xb = num(b); columns = []; counts = []
    for k, frame in sorted(cs.items()):
        keep = np.asarray([v not in bgkeys for v in keys(frame)])
        fallback = bool(keep.sum() <= 50)
        if fallback: keep[:] = True
        d = NearestNeighbors(n_neighbors=1).fit(num(frame[keep])).kneighbors(xb)[0][:, 0]
        columns.append(d); counts.append(dict(target=k, C_rows=len(frame), retained_rows=int(keep.sum()), fallback_all=fallback))
    ld = np.log(np.column_stack(columns)+1e-3)
    return np.median(ld, 1, keepdims=True)-ld, ld, counts


def age_mask(q, evidence, baseline, width):
    """Identical FA100 quota/tie arithmetic, extracted from preregistered proxy."""
    mask=np.zeros(len(q),bool);groups=[]
    ids=q.challenge_row_id.to_numpy();keys=list(q[QI].itertuples(index=False,name=None))
    bins=np.floor(q.age.to_numpy(float)/width).astype(int)
    for sex in sorted(q.sex.unique()):
        ix=np.flatnonzero(q.sex.to_numpy()==sex);quota=int(baseline[ix].sum());taken=0;nexts=[];cells={}
        for i in ix:cells.setdefault((str(q.prefecture.iloc[i]),int(bins[i])),[]).append(int(i))
        for key,cell in sorted(cells.items()):
            order=sorted(cell,key=lambda i:(-float(evidence[i]),keys[i],int(ids[i])))
            half=len(order)//2;mask[order[:half]]=True;taken+=half
            if len(order)%2:nexts.append(order[half])
            groups.append(dict(sex=str(sex),prefecture=key[0],age_bin=key[1],indices=cell))
        remaining=quota-taken
        if not 0<=remaining<=len(nexts):raise ValueError('Sex quota incompatible with half-cell allocation')
        nexts.sort(key=lambda i:(-float(evidence[i]),int(ids[i])))
        mask[nexts[:remaining]]=True
        assert int(mask[ix].sum())==quota
    return mask,groups
