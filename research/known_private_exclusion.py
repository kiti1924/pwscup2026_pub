"""Match own public challenge QI to own B; no outcome/membership inputs."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

NUM=['age','SBP','TG','HDL','ALT'];QI=NUM+['sex','prefecture']


def match(b,q,tolerant=False):
    if set(b)!=set(['record_id']+QI) or set(q)!=set(['challenge_row_id']+QI):
        raise ValueError('Only identifiers and the seven public QI are allowed')
    if b.record_id.duplicated().any() or q.challenge_row_id.duplicated().any():raise ValueError('Duplicate IDs')
    x=b[NUM].to_numpy(float);z=q[NUM].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(z).all() or b[QI].isna().any().any() or q[QI].isna().any().any():raise ValueError('Invalid QI')
    exact=(z[:,None,:]==x[None,:,:]).all(2)
    close=np.isclose(z[:,None,:],x[None,:,:],rtol=1e-12,atol=1e-10).all(2)
    category=(q.sex.to_numpy()[:,None]==b.sex.to_numpy()[None,:])&(q.prefecture.to_numpy()[:,None]==b.prefecture.to_numpy()[None,:])
    matrix=(close if tolerant else exact)&category
    unique=matrix&(matrix.sum(1)[:,None]==1)&(matrix.sum(0)[None,:]==1)
    qi,bi=np.nonzero(matrix)
    pairs=pd.DataFrame({'challenge_row_id':q.challenge_row_id.to_numpy()[qi],
        'record_id':b.record_id.to_numpy()[bi],'query_candidate_count':matrix.sum(1)[qi],
        'B_candidate_count':matrix.sum(0)[bi],'accepted':unique[qi,bi],
        'maximum_absolute_difference':np.max(np.abs(z[qi]-x[bi]),axis=1) if len(qi) else np.array([],float)})
    return unique.any(0),matrix,pairs


def exclude(values,mask):
    values=np.asarray(values,float);mask=np.asarray(mask,bool)
    if values.ndim!=2 or len(mask)!=len(values) or not np.isfinite(values).all() or mask.all():raise ValueError('Invalid input')
    result=np.zeros_like(values);n=int((~mask).sum())
    for k in range(values.shape[1]):result[~mask,k]=(rankdata(values[~mask,k],method='average')-.5)/n
    return result


def fixture(seed):
    rng=np.random.default_rng(seed)
    b=pd.DataFrame(rng.normal(size=(40,5)),columns=NUM)
    b['sex']=rng.integers(0,2,40);b['prefecture']=['p'+str(v) for v in rng.integers(0,3,40)]
    b.insert(0,'record_id',np.arange(40))
    q=b.iloc[:16][QI].copy().reset_index(drop=True);q.loc[8:,NUM]+=10
    q.insert(0,'challenge_row_id',np.arange(100,116))
    return b,q,np.arange(40)<8
