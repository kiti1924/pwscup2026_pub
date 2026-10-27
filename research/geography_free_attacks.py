"""Lawful-input geography-free predictions; truth is used only by the runner."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from research.adaptive_privacy import NUMERIC, DIST, digest, dump, matrix, qi_transform
from research.privacy_suite import QI, modal_window, mia_metrics, success_metrics, standardized_aia
from research.background_partition import SeparatedFold
from research.evaluate_background_transport import anchors

OUT=ROOT/'research/runs/20260925-G20-001'
VERSION='G17h-geography-free-v1'
MIA_COLS=NUMERIC+['sex','smoking','detected_rare']

def groups(q, d, rare):
    cols=['sex','smoking','detected_rare'] if rare else ['sex']
    mask=np.ones(len(d),bool)
    for col in cols: mask &= d[col].to_numpy()==q[col]
    if not mask.any(): mask=d.sex.to_numpy()==q['sex']
    return np.flatnonzero(mask) if mask.any() else np.arange(len(d))

def mia_predict(c, query, reference):
    if list(query.columns)!=MIA_COLS: raise ValueError('MIA requires public static columns only')
    scale=np.maximum(reference[NUMERIC].std(ddof=0).to_numpy(),1e-12)
    distances=[]
    for donors in [c,reference]:
        d=cdist(query[NUMERIC].to_numpy()/scale,donors[NUMERIC].to_numpy()/scale)
        distances.append(np.array([d[i,groups(q,donors,True)].min() for i,q in query.iterrows()]))
    dc,dr=distances
    return {'no_geo_conditional_nn':-dc,'no_geo_relative_nn':np.log(dr+1e-9)-np.log(dc+1e-9)}

def aia_predict(c, query, reference):
    if list(query.columns)!=QI: raise ValueError('AIA requires only lawful public QI')
    scale=np.maximum(qi_transform(reference).std(axis=0,ddof=1),1e-12)
    d=cdist(qi_transform(query)/scale,qi_transform(c)/scale,'sqeuclidean')
    nearest,modal=[],[]
    for i,q in query.iterrows():
        ix=groups(q,c,False);ds=d[i,ix];ts=c.time.to_numpy()[ix]
        nearest.append(float(np.sort(ts[ds==ds.min()]).mean()))
        threshold=np.partition(ds,min(20,len(ds))-1)[min(20,len(ds))-1]
        modal.append(modal_window(ts[ds<=threshold]))
    return {'no_geo_sex_1nn':np.array(nearest),'no_geo_sex_window20':np.array(modal)}

def evaluate(fold,path,out):
    from experiments.local_privacy_audit import _rare_mask_from_reference
    rel=path.relative_to(ROOT).as_posix()
    tag=path.stem+'_'+hashlib.sha256(rel.encode()).hexdigest()[:8]+f'_fold{fold.seed}'
    dest=out/(tag+'.json')
    if dest.exists():
        saved=json.loads(dest.read_text(encoding='utf-8'))
        assert saved['script_sha256']==digest(Path(__file__)) and saved['candidate_sha256']==digest(path)
        return saved
    c=pd.read_csv(path); ref=fold.ref.copy()
    c['detected_rare']=_rare_mask_from_reference(c,DIST)
    ref['detected_rare']=_rare_mask_from_reference(ref,DIST)
    mp=mia_predict(c,fold.qstatic[MIA_COLS],ref)
    ap=aia_predict(c,fold.query[QI],ref)
    # Truth begins here, after all predictions have been generated.
    result={'candidate':rel,'candidate_sha256':digest(path),'fold':fold.seed,'attack_version':VERSION,
        'script_sha256':digest(Path(__file__)),'split_sha256':fold.split_hash,'mia':{},'aia':{}}
    pred=pd.DataFrame({'record_id':fold.query.record_id,'source':np.where(fold.labels==1,'B14_member','background_control')})
    masks={'all':np.ones(len(fold.query),bool),'true_member_rare_detected_control':np.r_[fold.mr,fold.cr]}
    for k,v in mp.items():
        result['mia'][k]={g:mia_metrics(v,fold.labels,m) for g,m in masks.items()}
        pred['mia_'+k]=v
    for k,v in ap.items():
        hit=abs(v-fold.query.time.to_numpy())<=.5;mh,ch=hit[:len(fold.b)],hit[len(fold.b):]
        result['aia'][k]={'all':success_metrics(mh,ch),'rare_proxy':success_metrics(mh[fold.mr],ch[fold.cr]),
                         'stratified_all':standardized_aia(fold.b,fold.controls,mh,ch)}
        pred['time_hat_'+k]=v
    pred.to_csv(out/(tag+'_predictions.csv'),index=False,lineterminator='\n')
    dump(dest,result)
    print(f'{fold.seed} {path.name} geography-free saved',flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);args=ap.parse_args()
    shortlist=json.loads((OUT/'shortlist.json').read_text(encoding='utf-8'))
    paths=[ROOT/p for p in shortlist['candidates']]
    paths += [p for k,p in anchors().items() if k.startswith('positive_') or k.startswith('negative_')]
    out=OUT/'geography_evaluation';out.mkdir(exist_ok=True)
    fold=SeparatedFold(args.fold)
    results=[evaluate(fold,p,out) for p in paths]
    frame=matrix(results);frame['version']=VERSION
    frame.to_csv(OUT/f'geography_fold{args.fold}_matrix.csv',index=False,lineterminator='\n')
    raw=next(r for r in results if r['candidate'].endswith('C_positive_raw.csv'))
    gates={**{k:v['all']['tpr_at_fpr_0.01']>=.9 for k,v in raw['mia'].items()},
           **{k:v['all']['r_raw']>=.8 for k,v in raw['aia'].items()}}
    dump(OUT/f'geography_fold{args.fold}_checkpoint.json',{'positive_raw_gates':gates,'version':VERSION,
         'inputs':{str(p.relative_to(ROOT)):digest(p) for p in paths},'n_evaluations':len(results)})
    print('RAW CONTROL GATES',gates,flush=True)

if __name__=='__main__':main()
