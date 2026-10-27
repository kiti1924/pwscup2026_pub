"""FA03 pure lawful-input algorithms: background-excluded peers and local time mass."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from research.final_attack_api import (STATIC, NUM, QI, COLUMNS, frame_check, checked_keys,
    rank_unit, block, log_transform)

VERSION='FA03-lawful-v1'
AIA_METHODS=['weighted_window5','contrast_window5']


def static_keys(frame):
    frame=frame[STATIC].copy()
    for col in NUM+['sex','smoking']:frame[col]=frame[col].astype(float)
    return pd.util.hash_pandas_object(frame.round(10),index=False).to_numpy()


def conditional_distance(q,d,scale):
    if not len(d):return np.full(len(q),np.nan)
    distances=cdist(q[NUM].to_numpy()/scale,d[NUM].to_numpy()/scale,'sqeuclidean')
    for col in ['sex','smoking','prefecture']:
        distances+=2*(q[col].to_numpy()[:,None]!=d[col].to_numpy()[None,:])
    distances=np.sqrt(distances);out=[]
    for i,row in enumerate(q.itertuples(index=False)):
        sex=d.sex.to_numpy()==row.sex;geo=sex&(d.prefecture.to_numpy()==row.prefecture)
        band=geo&(np.floor(d.age.to_numpy()/10)==np.floor(row.age/10))
        mask=next(m for m in [band,geo,sex,np.ones(len(d),bool)] if m.any())
        out.append(distances[i,mask].min())
    return np.asarray(out)


def predict_peer_batch(cs,query,background,own_team):
    frame_check(query,['record_id']+STATIC);checked_keys(query,'record_id')
    frame_check(background,COLUMNS)
    q=query[STATIC];bg=background[STATIC]
    scale=np.maximum(bg[NUM].std(ddof=0).to_numpy(),1e-12);bk=set(static_keys(bg))
    distances={};counts=[]
    for k,c in cs.items():
        frame_check(c,COLUMNS)
        if k==own_team:continue
        mask=~np.isin(static_keys(c),list(bk));clean=c.loc[mask,STATIC]
        distances[k]=conditional_distance(q,clean,scale)
        counts.append({'target':k,'clean_rows':len(clean),'removed':len(c)-len(clean)})
    fallback=conditional_distance(q,bg,scale);result={}
    for k,dc in distances.items():
        peers=[v for j,v in distances.items() if j!=k and np.isfinite(v).any()]
        dp=np.median(np.column_stack(peers),axis=1) if peers else fallback
        raw=np.zeros(len(q)) if np.isnan(dc).all() else np.log(dp+1e-9)-np.log(dc+1e-9)
        result[k]=pd.DataFrame({'record_id':query.record_id.to_numpy(),'peer_bg_excluded':rank_unit(raw)})
    return result,pd.DataFrame(counts)


def neighbours(distances,times,indices,k):
    d=distances[indices];bound=np.partition(d,min(k,len(d))-1)[min(k,len(d))-1]
    keep=d<=bound;ts=times[indices][keep];dd=d[keep]
    order=np.lexsort((dd,ts));weights=1/(dd[order]+1e-9);weights/=weights.sum()
    return ts[order],weights


def time_window(times,weights,reference=None):
    ts=np.sort(times);ends=np.searchsorted(ts,ts+1,side='right')
    centers=(ts+ts[np.maximum(ends-1,0)])/2
    mass=np.array([weights[np.abs(times-center)<=.5].sum() for center in centers])
    if reference is not None:
        rt,rw=reference
        mass-=np.array([rw[np.abs(rt-center)<=.5].sum() for center in centers])
    best=int(np.argmax(mass))
    return float(centers[best]),float(mass[best])


def predict_local_aia(c,query,background):
    frame_check(c,COLUMNS);frame_check(background,COLUMNS)
    frame_check(query,['challenge_row_id']+QI);checked_keys(query,'challenge_row_id')
    scale=np.maximum(log_transform(background).std(axis=0,ddof=1),1e-12)
    ds=[cdist(log_transform(query)/scale,log_transform(d)/scale,'sqeuclidean') for d in [c,background]]
    sentinel=float(max(c.time.max(),background.time.max())+2);weighted=[];contrast=[]
    for i,row in enumerate(query[QI].itertuples(index=False)):
        ct,cw=neighbours(ds[0][i],c.time.to_numpy(),block(row,c),5)
        rt,rw=neighbours(ds[1][i],background.time.to_numpy(),block(row,background),20)
        weighted.append(time_window(ct,cw)[0]);prediction,mass=time_window(ct,cw,(rt,rw))
        contrast.append(prediction if mass>0 else sentinel)
    return pd.DataFrame({'challenge_row_id':query.challenge_row_id.to_numpy(),
        'weighted_window5':weighted,'contrast_window5':contrast})
