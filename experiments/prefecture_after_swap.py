"""Prefecture overlay audit after outcome-swap hillclimb."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
STARTER=ROOT/'pwscup2026-kit'/'starter'
SRC=STARTER/'src'
sys.path.insert(0,str(STARTER)); sys.path.insert(0,str(SRC))
os.environ['PYTHONPATH']=os.pathsep.join([str(SRC),str(STARTER),os.environ.get('PYTHONPATH','')])
from pwscup2026.scoring import attack_score

DIST=ROOT/'_private_team10'
BASE=DIST/'calibrated_tilt030_final_fine'/'C_cal_tilt030_finalfine_u15_h1.000_lr0.350_i7.csv'
CURRENT=DIST/'outcome_swap_hillclimb'/'final_best.csv'
MOVES=DIST/'prefecture_extended_best0350'/'moves.json'
OUT=DIST/'prefecture_after_swap'
NUM=['age','BMI','SBP','TG','HDL','ALT','FPG']; CAT=['sex','smoking','prefecture']
STD=["age","BMI","SBP","TG","HDL","ALT","FPG"]

def overlay(cur,c0,moves,ks):
    p=c0['prefecture'].copy()
    for m in moves[:ks]: p.iloc[m['row_index']]=m['to']
    z=cur.copy(deep=True); z['prefecture']=p.to_numpy(); return z

def distmat(x,c,mu,sd):
    xn=(x[NUM].to_numpy(float)-mu)/sd; cn=(c[NUM].to_numpy(float)-mu)/sd
    d=((xn[:,None,:]-cn[None,:,:])**2).sum(2)
    for col in CAT: d += 2*(x[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
    return np.sqrt(d).min(1)

def static_one(c,b,controls,rare,ctrl_rare,mu,sd):
    dm=distmat(b,c,mu,sd); dc=distmat(controls,c,mu,sd)
    conf=-np.r_[dm,dc]; truth=np.r_[np.ones(len(b),int),np.zeros(len(controls),int)]
    tpr,auc=attack_score._tpr_at_fpr(conf,truth,.01)
    rm=np.r_[rare,ctrl_rare]; rtpr,_=attack_score._tpr_at_fpr(conf[rm],truth[rm],.01)
    exact_cols=['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
    s=set(c[exact_cols].itertuples(index=False,name=None))
    me=sum(tuple(r) in s for r in b[exact_cols].itertuples(index=False,name=None)); ce=sum(tuple(r) in s for r in controls[exact_cols].itertuples(index=False,name=None))
    return {'all_tpr_at_fpr_0.01':float(tpr),'true_rare_tpr_at_fpr_0.01':float(rtpr),'roc_auc':float(auc),'member_d_le_0.5':int((dm<=.5).sum()),'control_d_le_0.5':int((dc<=.5).sum()),'member_exact_match_13col':int(me),'control_exact_match_13col':int(ce)}

def score(path):
    py=STARTER/'.venv'/'Scripts'/'python.exe'
    p=subprocess.run([str(py),'-m','pwscup2026.kit.selfscore',str(path),'--dist',str(DIST),'--json'],cwd=STARTER,capture_output=True,text=True,check=False,env=os.environ.copy())
    if p.returncode: raise RuntimeError(p.stderr[-1000:])
    return json.loads(p.stdout)

def audit(script,paths,out):
    py=STARTER/'.venv'/'Scripts'/'python.exe'
    p=subprocess.run([str(py),str(script),'--dist',str(DIST),'--c',*map(str,paths),'--out',str(out)],cwd=STARTER,capture_output=True,text=True,check=False,env=os.environ.copy())
    if p.returncode: raise RuntimeError(p.stderr[-1000:])
    return json.loads(out.read_text(encoding='utf-8'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c0=pd.read_csv(BASE); cur=pd.read_csv(CURRENT); moves=json.loads(MOVES.read_text(encoding='utf-8'))
    assert list(c0.columns)==list(cur.columns) and len(cur)==1049
    assert np.array_equal(cur.drop(columns=['prefecture']).astype(str).to_numpy(),cur.drop(columns=['prefecture']).astype(str).to_numpy())
    b=pd.read_csv(DIST/'B_practice.csv'); a=pd.read_csv(DIST/'A_bg.csv')
    controls=a.iloc[np.random.default_rng(0).choice(len(a),1049,replace=False)].reset_index(drop=True)
    mu=a[NUM].mean().to_numpy(); sd=a[NUM].std(ddof=0).to_numpy(); sd=np.where(sd>0,sd,1)
    ref=json.loads((DIST/'utility_ref.json').read_text(encoding='utf-8'))
    rare=pd.read_csv(DIST/'B_self.csv')['is_rare'].to_numpy(bool)
    ctrl_rare=(__import__('pwscup2026.rare.outlier',fromlist=['mahalanobis']).mahalanobis(controls,np.asarray(ref['mu'],float),np.asarray(ref['sigma'],float),NUM)>=float(ref['band']['lo']))
    # Verify the known prefix-12 overlay against current best.
    p12=overlay(cur,c0,moves,12); assert np.array_equal(p12['prefecture'].to_numpy(),cur['prefecture'].to_numpy())
    candidates={}
    for k in [32,33,34,40]: candidates[f'C_pref_actual_k{k}.csv']=overlay(cur,c0,moves,k)
    # Prefix-12 baseline for privacy-aware single-move ranking.
    base12=overlay(cur,c0,moves,12); base12_static=static_one(base12,b,controls,rare,ctrl_rare,mu,sd)
    impacts=[]
    for i,m in enumerate(moves[12:],start=13):
        q=base12.copy(); q.loc[m['row_index'],'prefecture']=m['to']
        s=static_one(q,b,controls,rare,ctrl_rare,mu,sd)
        impacts.append({'move_index':i,'row_index':m['row_index'],'from':m['from'],'to':m['to'],'auc':s['roc_auc'],'auc_delta':s['roc_auc']-base12_static['roc_auc'],'all_tpr':s['all_tpr_at_fpr_0.01']})
    selected=sorted(impacts,key=lambda x:(x['auc_delta'],x['auc'],x['move_index']))[:21]
    q=base12.copy()
    for x in selected: q.loc[x['row_index'],'prefecture']=x['to']
    candidates['C_pref_privacyaware_21.csv']=q
    paths=[]; scores={}
    for name,c in candidates.items():
        path=OUT/name; c.to_csv(path,index=False); paths.append(path); scores[name]=score(path)
        (OUT/(name.replace('.csv','.score.json'))).write_text(json.dumps(scores[name],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fs=audit(ROOT/'experiments'/'full_static_nearest_audit.py',paths,OUT/'full_static.json'); lp=audit(ROOT/'experiments'/'local_privacy_audit.py',paths,OUT/'privacy.json')
    fsm={x['candidate']:x for x in fs['candidates']}; lpm={x['candidate']:x for x in lp['candidates']}
    rows=[]
    for name in candidates:
        f=fsm[name]; l=lpm[name]; sc=scores[name]; fail=[]
        if f['all_tpr_at_fpr_0.01']>.033365109628+1e-12: fail.append('allTPR')
        if f['true_rare_tpr_at_fpr_0.01']>1/24+1e-12: fail.append('rareTPR')
        if f['roc_auc']>.578820811686+1e-12: fail.append('AUC')
        if f['member_d_le_0.5']>4 or f['control_d_le_0.5']!=0: fail.append('distance')
        if f['member_exact_match_13col'] or f['control_exact_match_13col']: fail.append('exact13')
        if l['mia']['exposure']>.3333334+1e-12: fail.append('MIA')
        if abs(l['aia']['attack_score'])>1e-12: fail.append('AIA')
        if sc['facets']['U_rare']<.9780967702-1e-12 or sc['facets']['U_valid']!=1: fail.append('utility')
        rows.append({'candidate':name,'passes':not fail,'fail':fail,'Ugen':sc['facets']['U_gen'],'Urare':sc['facets']['U_rare'],'Uvalid':sc['facets']['U_valid'],'full_static':f,'privacy':l})
    report={'base':str(BASE),'current':str(CURRENT),'prefix12_static':base12_static,'selected_moves':selected,'single_move_impacts':impacts,'results':rows}
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in rows: print(r['candidate'], 'PASS' if r['passes'] else 'FAIL '+','.join(r['fail']), 'Ugen=',r['Ugen'],'AUC=',r['full_static']['roc_auc'])
    return 0
if __name__=='__main__': raise SystemExit(main())
