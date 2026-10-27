"""FA35 pure public C/reference classifier, query QI only."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

NUM=['age','SBP','TG','HDL','ALT'];QI=NUM+['sex','prefecture']
SEED=2026101039


def query_check(q):
    if set(q.columns)-set(QI+['challenge_row_id']):raise ValueError('Secret/extra query fields rejected')
    if not set(QI).issubset(q.columns):raise ValueError('Missing query QI')
    assert q.challenge_row_id.is_unique


def frame(data,county):
    out=data[NUM].astype(float).copy()
    for n in ['TG','ALT']:out[n]=np.log1p(out[n])
    for n in ['sex']+(['prefecture'] if county else []):out[n]=data[n].astype(str)
    if not np.isfinite(out[NUM].to_numpy()).all():raise ValueError('Nonfinite numeric QI')
    return out


def prepare(c,bg,county):
    if not len(c) or not len(bg):raise ValueError('Both public training sources required')
    cc=frame(c,county).reset_index(drop=True);bb=frame(bg,county).reset_index(drop=True)
    cc['_source']=1;bb['_source']=0;cc['_index']=np.arange(len(c));bb['_index']=np.arange(len(bg))
    rows=pd.concat([cc,bb],ignore_index=True).sort_values(list(frame(c,county).columns)+['_source'],kind='mergesort').reset_index(drop=True)
    cat=['sex']+(['prefecture'] if county else [])
    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    x=np.column_stack([rows[NUM].to_numpy(),encoder.fit_transform(rows[cat])])
    y=rows['_source'].to_numpy(int);weight=np.where(y==1,len(y)/(2*len(c)),len(y)/(2*len(bg)))
    return {'x':x,'y':y,'weight':weight,'source_index':rows['_index'].to_numpy(int)},encoder


def fit(c,bg,county,family):
    data,encoder=prepare(c,bg,county)
    if family=='hgb':model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=SEED)
    elif family=='extra':model=ExtraTreesClassifier(n_estimators=160,max_depth=12,min_samples_leaf=3,max_features=.7,n_jobs=1,random_state=SEED)
    else:raise ValueError(family)
    model.fit(data['x'],data['y'],sample_weight=data['weight'])
    return {'model':model,'encoder':encoder,'county':county,'family':family,'seed':SEED,'n_C':len(c),'n_BG':len(bg),
        'teacher_used':False,'query_truth_used':False},data


def predict(asset,q):
    query_check(q);f=frame(q,asset['county']);cat=['sex']+(['prefecture'] if asset['county'] else [])
    x=np.column_stack([f[NUM].to_numpy(),asset['encoder'].transform(f[cat])])
    return asset['model'].predict_proba(x)[:,1]


def select_half(prob,q):
    query_check(q);groups={s:np.flatnonzero(q.sex.to_numpy()==s) for s in sorted(q.sex.unique())}
    K=len(q)//2;quota={s:int(np.floor(K*len(ix)/len(q))) for s,ix in groups.items()}
    order=sorted(groups,key=lambda s:(-(K*len(groups[s])/len(q)-quota[s]),s))
    for s in order[:K-sum(quota.values())]:quota[s]+=1
    keys=list(q[QI].itertuples(index=False,name=None));ids=q.challenge_row_id.to_numpy();mask=np.zeros(len(q),bool)
    for s,ix in groups.items():
        ordered=sorted(ix,key=lambda i:(-prob[i],keys[i],ids[i]));mask[ordered[:quota[s]]]=True
    assert mask.sum()==K
    return mask
