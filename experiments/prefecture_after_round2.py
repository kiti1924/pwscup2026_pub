"""Greedy privacy-aware extension after outcome-swap round 2."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; STARTER=ROOT/'pwscup2026-kit'/'starter'; SRC=STARTER/'src'
sys.path.insert(0,str(STARTER)); sys.path.insert(0,str(SRC)); os.environ['PYTHONPATH']=os.pathsep.join([str(SRC),str(STARTER),os.environ.get('PYTHONPATH','')])
from pwscup2026.scoring import attack_score
from prefecture_after_swap import score, audit

DIST=ROOT/'_private_team10'; BASE=DIST/'outcome_swap_hillclimb_round2'/'final_best.csv'; MOVES=DIST/'prefecture_extended_best0350'/'moves.json'; PREV=DIST/'prefecture_after_swap'/'summary.json'; OUT=DIST/'prefecture_after_round2'
NUM=['age','BMI','SBP','TG','HDL','ALT','FPG']; CAT=['sex','smoking','prefecture']; CHECK_AUC=.578820811686; TARGET_UGEN=.97630950002

def dmin(x,c,mu,sd):
    xn=(x[NUM].to_numpy(float)-mu)/sd; cn=(c[NUM].to_numpy(float)-mu)/sd; d=((xn[:,None,:]-cn[None,:,:])**2).sum(2)
    for col in CAT: d += 2*(x[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
    return np.sqrt(d).min(1)
def static(c,b,ctl,rare,cr,mu,sd):
    dm=dmin(b,c,mu,sd); dc=dmin(ctl,c,mu,sd); conf=-np.r_[dm,dc]; truth=np.r_[np.ones(len(b),int),np.zeros(len(ctl),int)]
    tpr,auc=attack_score._tpr_at_fpr(conf,truth,.01); rm=np.r_[rare,cr]; rtpr,_=attack_score._tpr_at_fpr(conf[rm],truth[rm],.01)
    cols=['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']; s=set(c[cols].itertuples(index=False,name=None)); me=sum(tuple(r) in s for r in b[cols].itertuples(index=False,name=None)); ce=sum(tuple(r) in s for r in ctl[cols].itertuples(index=False,name=None))
    return {'all':float(tpr),'rare':float(rtpr),'auc':float(auc),'member':int((dm<=.5).sum()),'control':int((dc<=.5).sum()),'exact_m':int(me),'exact_c':int(ce)}
def ok(s): return s['all']<=.033365109628+1e-12 and s['rare']<=1/24+1e-12 and s['auc']<=CHECK_AUC+1e-12 and s['member']<=4 and s['control']==0 and s['exact_m']==0 and s['exact_c']==0
def main():
    OUT.mkdir(parents=True,exist_ok=True); cur=pd.read_csv(BASE); moves=json.loads(MOVES.read_text(encoding='utf-8')); prev=json.loads(PREV.read_text(encoding='utf-8')); selected={int(x['row_index']) for x in prev['selected_moves']}
    assert len(moves)==62 and len(selected)==21
    b=pd.read_csv(DIST/'B_practice.csv'); a=pd.read_csv(DIST/'A_bg.csv'); ctl=a.iloc[np.random.default_rng(0).choice(len(a),1049,replace=False)].reset_index(drop=True); mu=a[NUM].mean().to_numpy(); sd=np.where(a[NUM].std(ddof=0).to_numpy()>0,a[NUM].std(ddof=0).to_numpy(),1)
    ref=json.loads((DIST/'utility_ref.json').read_text(encoding='utf-8')); from pwscup2026.rare import outlier; cr=outlier.mahalanobis(ctl,np.asarray(ref['mu'],float),np.asarray(ref['sigma'],float),NUM)>=float(ref['band']['lo']); rare=pd.read_csv(DIST/'B_self.csv')['is_rare'].to_numpy(bool)
    remaining=[m for m in moves[12:] if int(m['row_index']) not in selected]; assert len(remaining)==29
    # The round2 base already includes prefix12 + previous selected21.
    step_rows=[]; states=[]; current=cur.copy(); current_static=static(current,b,ctl,rare,cr,mu,sd); assert ok(current_static)
    base_ugen=.97521448999; delta=1/(1049*10)
    for step in range(1,30):
        trials=[]
        for m in remaining:
            q=current.copy(); i=int(m['row_index']); assert str(q.at[i,'prefecture'])==m['from']; q.at[i,'prefecture']=m['to']; ss=static(q,b,ctl,rare,cr,mu,sd)
            if ok(ss): trials.append((ss['auc'],int(m['row_index']),m,q,ss))
        if not trials: break
        _,_,m,current,current_static=min(trials,key=lambda z:(z[0],z[1])); remaining=[x for x in remaining if int(x['row_index'])!=int(m['row_index'])]
        ugen=base_ugen+step*delta; step_rows.append({'step':step,'move_index':moves.index(m)+1,'row_index':m['row_index'],'from':m['from'],'to':m['to'],'Ugen_est':ugen,'static':current_static}); states.append(current.copy())
    need=next((x['step'] for x in step_rows if x['Ugen_est']>=TARGET_UGEN),None); assert need is not None
    min_c=states[need-1]; max_c=states[-1]; min_name='C_pref_round2_min12.csv'; max_name='C_pref_round2_maxsafe.csv'; min_path=OUT/min_name; max_path=OUT/max_name; min_c.to_csv(min_path,index=False); max_c.to_csv(max_path,index=False)
    paths=[min_path,max_path]; sc={p.name:score(p) for p in paths}; fs=audit(ROOT/'experiments'/'full_static_nearest_audit.py',paths,OUT/'full_static.json'); lp=audit(ROOT/'experiments'/'local_privacy_audit.py',paths,OUT/'privacy.json'); fm={x['candidate']:x for x in fs['candidates']}; lm={x['candidate']:x for x in lp['candidates']}
    results=[]
    for p in paths:
        f=fm[p.name]; l=lm[p.name]; results.append({'candidate':p.name,'add_count':need if p==min_path else len(step_rows),'score':sc[p.name],'full_static':f,'privacy':l,'passes':f['all_tpr_at_fpr_0.01']<=.033365109628+1e-12 and f['true_rare_tpr_at_fpr_0.01']<=1/24+1e-12 and f['roc_auc']<=CHECK_AUC+1e-12 and f['member_d_le_0.5']<=4 and f['control_d_le_0.5']==0 and f['member_exact_match_13col']==0 and f['control_exact_match_13col']==0 and l['mia']['exposure']<=.3333334+1e-12 and abs(l['aia']['attack_score'])<=1e-12 and sc[p.name]['facets']['U_valid']==1})
    report={'base':str(BASE),'remaining_count':29,'step_rows':step_rows,'minimum_count_to_target':need,'results':results,'selected_rows_previous':sorted(selected)}; (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in results: print(r['candidate'],'PASS' if r['passes'] else 'FAIL','add',r['add_count'],'Ugen',r['score']['facets']['U_gen'],'AUC',r['full_static']['roc_auc'],'MIA',r['privacy']['mia']['exposure'],'AIA',r['privacy']['aia']['attack_score'])
if __name__=='__main__': main()
