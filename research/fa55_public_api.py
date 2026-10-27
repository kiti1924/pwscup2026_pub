"""FA95: pure public-frame FA55 mean; no IO, truth, teacher or scoring imports."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import distance_norms as l1
import trimmed_distance as trim
import selection_ensemble as ensemble

VERSION='FA95-FA55-public-v1'
NUM=['age','SBP','TG','HDL','ALT']
QI=NUM+['sex','prefecture']
PUBLIC=l1.base.PUBLIC
STATIC=l1.base.SEVEN+['sex','smoking','prefecture']


def keys(frame,key,expected=None):
    v=frame[key].to_numpy(float)
    if not np.isfinite(v).all() or not np.equal(v,np.floor(v)).all() or frame[key].duplicated().any():
        raise ValueError('Keys must be finite integral and unique')
    if expected is not None and set(frame[key])!=set(expected):
        raise ValueError('Missing or extra keys')


def check(frame,query=False,expected=None):
    required=set(QI+['challenge_row_id'] if query else PUBLIC)
    allowed=required if query else required|{'record_id'}
    if not frame.columns.is_unique or not required.issubset(frame) or set(frame)-allowed:
        raise ValueError('Missing, secret, duplicate or unknown columns')
    numeric=list(required-{'prefecture'})
    if not len(frame) or frame.isna().any().any() or not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError('Empty or nonfinite public input')
    if query:keys(frame,'challenge_row_id',expected)
    elif 'record_id' in frame:keys(frame,'record_id')


def canonical(frame):
    check(frame)
    return frame[PUBLIC].sort_values(PUBLIC,kind='mergesort').reset_index(drop=True)


def prepare(background,candidates,expected_pool):
    if set(candidates)!=set(expected_pool) or len(candidates)<2:
        raise ValueError('Incomplete public rival pool')
    check(background)
    # Public key ordering reproduces original reductions and is invariant to row shuffle.
    bg=(background.sort_values('record_id',kind='mergesort')[PUBLIC].reset_index(drop=True)
        if 'record_id' in background else canonical(background))
    asset=l1.fit(bg,'l1_log5')
    cs={int(k):canonical(candidates[k]) for k in sorted(candidates)}
    return dict(version=VERSION,background=bg,scale=asset,bg_x=l1.transform(bg,asset),
                candidates=cs,c_x={k:l1.transform(c,asset) for k,c in cs.items()},pool=list(cs))


def predict(asset,query,target,expected_ids,soft=None):
    check(query,True,expected_ids)
    if target not in asset['candidates']:raise ValueError('Target not in public pool')
    q=query.reset_index(drop=True)
    x=l1.transform(q,asset['scale'],aia=True)
    cx=asset['c_x'][target];c=asset['candidates'][target]
    dl,donor,_=l1.nearest(x,cx,q.sex.to_numpy(),c.sex.to_numpy(),False,'cityblock')
    db,_,_=l1.nearest(x,asset['bg_x'],q.sex.to_numpy(),asset['background'].sex.to_numpy(),False,'cityblock')
    dc=[]
    for k in asset['pool']:
        cc=asset['candidates'][k]
        d,_,_=trim.nearest(x,asset['c_x'][k],q.sex.to_numpy(),cc.sex.to_numpy(),False,'trimmed')
        dc.append(d)
    dc=np.column_stack(dc);j=asset['pool'].index(target)
    rival=np.delete(np.log(dc+.001),j,axis=1).min(1)
    v1=np.log(db+.001)-np.log(dl+.001);v2=rival-np.log(dc[:,j]+.001)
    donor_time=c.time.to_numpy()[donor]
    m1=ensemble.d.select_half(v1,q);m2=ensemble.d.select_half(v2,q)
    if soft is None:soft=np.zeros(len(q))
    pred,masks,ranks,mean=ensemble.combine(q,v1,v2,m1,m2,donor_time,np.asarray(soft))
    dd=cdist(x,cx,'cityblock');top2=np.partition(dd,1,axis=1)[:,:2];top2.sort(axis=1)
    return dict(challenge_row_id=q.challenge_row_id.to_numpy(),v1=v1,v2=v2,donor_time=donor_time,
                donor_canonical_index=donor,donor_public_rows=c.iloc[donor].to_numpy(),
                mask=masks['mean'],mask_l1=m1,mask_trim=m2,ranks=ranks,mean_evidence=mean,
                prediction=pred['mean'],l1_distance=dl,bg_distance=db,trim_distances=dc,
                nearest_gap=top2[:,1]-top2[:,0],pool=np.array(asset['pool']))


def fingerprint(asset,query,target,result):
    """FA54/FA12 definitions plus fixed summaries of the same public evidence."""
    c=asset['candidates'][target];bg=asset['background'];mask=result['mask']
    bgkeys=set(bg[STATIC].itertuples(index=False,name=None))
    row=dict(c_late_fraction=float(c.time.ge(9.5).mean()),c_time_unique=float(c.time.nunique()/len(c)),
             c_qi_unique=float(len(c[NUM+['sex']].drop_duplicates())/len(c)),
             static_duplicate=float(c[STATIC].duplicated().mean()),
             bg_static_match=float(np.mean([v in bgkeys for v in c[STATIC].itertuples(index=False,name=None)])),
             l1_target_p10=float(np.quantile(result['l1_distance'],.1)),
             l1_target_p50=float(np.quantile(result['l1_distance'],.5)),
             l1_target_p90=float(np.quantile(result['l1_distance'],.9)),
             l1_nearest_gap_p50=float(np.median(result['nearest_gap'])),
             l1_bg_ratio_p50=float(np.median(result['v1'])),trim_rival_ratio_p50=float(np.median(result['v2'])),
             selected_fraction=float(mask.mean()),
             selected_late_donor_fraction=float(np.mean(result['donor_time'][mask]>=9.5)),
             mask_disagreement=float(np.mean(result['mask_l1']!=result['mask_trim'])))
    sexes=[]
    for sex in sorted(query.sex.unique()):
        ix=query.sex.to_numpy()==sex
        sexes.append(dict(sex=int(sex),n=int(ix.sum()),selected=int(mask[ix].sum()),
                          selected_fraction=float(mask[ix].mean()),
                          selected_late_donor_fraction=float(np.mean(result['donor_time'][ix&mask]>=9.5))))
    return row,sexes
