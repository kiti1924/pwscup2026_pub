"""FA55 fixed legal-input rank and set mask combinations."""
import numpy as np
from scipy.stats import rankdata
import direct_density_aia as d
METHODS=['L1','trim','mean','intersection','union','L1_inter','trim_inter','L1_union','trim_union','soft1']
STRATEGIES=['L1','trim','mean','intersection','union','soft1']
CANDIDATES=['mean','intersection','union']
BASELINES=['L1','trim','soft1']
PAIRS=[('intersection','L1_inter'),('intersection','trim_inter'),('union','L1_union'),('union','trim_union')]

def select_count(v,q,template):
    d.query_check(q);v=np.asarray(v);template=np.asarray(template)
    if v.shape!=(len(q),) or template.shape!=v.shape or template.dtype!=bool or not np.isfinite(v).all():raise ValueError('Aligned finite evidence and boolean template required')
    keys=list(q[d.QI].itertuples(index=False,name=None));ids=q.challenge_row_id.to_numpy();out=np.zeros(len(q),bool)
    for sex in sorted(q.sex.unique()):
        ix=np.flatnonzero(q.sex.to_numpy()==sex);n=int(template[ix].sum());order=sorted(ix,key=lambda i:(-v[i],keys[i],ids[i]));out[order[:n]]=True
    return out

def combine(q,v1,v2,m1,m2,time,soft):
    d.query_check(q);vectors=[np.asarray(x) for x in [v1,v2,m1,m2,time,soft]]
    if any(v.shape!=(len(q),) for v in vectors):raise ValueError('Aligned vectors required')
    if any(v.dtype!=bool for v in vectors[2:4]):raise ValueError('Boolean masks required')
    if any(not np.isfinite(v).all() for v in vectors):raise ValueError('Finite vectors required')
    v1,v2,m1,m2,time,soft=vectors;ranks=np.empty((len(q),2))
    for sex in sorted(q.sex.unique()):
        ix=np.flatnonzero(q.sex.to_numpy()==sex)
        for j,v in enumerate([v1,v2]):ranks[ix,j]=rankdata(v[ix],method='average')/len(ix)
    mean=ranks.mean(1);masks=dict(L1=m1.copy(),trim=m2.copy(),mean=d.select_half(mean,q),intersection=m1&m2,union=m1|m2)
    for suffix,template in [('inter',masks['intersection']),('union',masks['union'])]:
        masks['L1_'+suffix]=select_count(v1,q,template);masks['trim_'+suffix]=select_count(v2,q,template)
    pred={n:np.where(masks[n],time,12.) for n in METHODS[:-1]};pred['soft1']=soft.copy()
    return pred,masks,ranks,mean
