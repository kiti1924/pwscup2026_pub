"""G05 detected-rare residual copula synthesis with fixed Cox calibration."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.run_processing import KIT
from research.adaptive_privacy import Fold, dump, digest
from research.repair_sex_correlation import validate_score, risk_gates
from experiments.run_aligned_synth import (_load_detector,_detector_mask,_format_continuous,
    _sample_categorical,_pseudo_observations,_nearest_correlation,_fit_cause_cox)
from experiments.run_calibrated_outcomes import _run_path
from reference._common import load_cfg, write_c
from pwscup2026.common import schema
import numpy as np
import pandas as pd
from scipy.stats import norm

DIST=ROOT/'_private_team14'
BASE=ROOT/'research/runs/20260924-G02-001/C_rare_aligned_s0.csv'
OUT=ROOT/'research/runs/20260924-G05-001'
XCOL=['age','BMI','SBP','TG','HDL','ALT']
STATIC=[*schema.CONTINUOUS,'sex','smoking','prefecture']


def predictors(frame):
    x=frame[XCOL].to_numpy(float).copy()
    x[:,[3,5]]=np.log1p(x[:,[3,5]])
    return x


def fit_residual(b,detector):
    rare=_detector_mask(b,*detector)
    x=predictors(b);mean=x[~rare].mean(axis=0);scale=x[~rare].std(axis=0)
    scale=np.maximum(scale,1e-9)
    design=np.column_stack([np.ones(len(b)),(x-mean)/scale])
    penalty=np.diag([0.,1.,1.,1.,1.,1.,1.])
    beta=np.linalg.solve(design[~rare].T@design[~rare]+penalty,design[~rare].T@b.FPG.to_numpy()[~rare])
    residual=b.FPG.to_numpy()-design@beta
    joint=np.column_stack([x[rare],residual[rare]])
    z=norm.ppf(np.clip(_pseudo_observations(joint),1e-6,1-1e-6))
    corr=_nearest_correlation(np.corrcoef(z,rowvar=False))
    return {'mean':mean,'scale':scale,'beta':beta,'joint':joint,'corr':corr}


def residual_values(frame,model):
    design=np.column_stack([np.ones(len(frame)),(predictors(frame)-model['mean'])/model['scale']])
    return frame.FPG.to_numpy()-design@model['beta']


def generate(base,b,cfg,detector,model,alpha,seed):
    rng=np.random.default_rng(5100+seed)
    rare=_detector_mask(base,*detector);n=int(rare.sum())
    corr=(1-alpha)*model['corr']+alpha*np.eye(7)
    accepted=[];count=0;proposed=0
    while count<n:
        size=max(128,(n-count)*4)
        if proposed+size>118000:
            raise RuntimeError('Rejection budget exhausted')
        u=norm.cdf(rng.multivariate_normal(np.zeros(7),corr,size=size))
        sampled=np.column_stack([np.quantile(model['joint'][:,j],u[:,j]) for j in range(7)])
        x=sampled[:,:6].copy()
        design=np.column_stack([np.ones(size),(x-model['mean'])/model['scale']])
        fpg=design@model['beta']+sampled[:,6]
        x[:,[3,5]]=np.expm1(x[:,[3,5]])
        frame=pd.DataFrame(x,columns=XCOL);frame['FPG']=fpg
        frame=_format_continuous(frame,cfg)
        keep=_detector_mask(frame,*detector)
        accepted.append(frame.loc[keep]);count+=int(keep.sum());proposed+=size
    rare_frame=pd.concat(accepted,ignore_index=True).iloc[:n].reset_index(drop=True)
    rare_frame=_sample_categorical(b,rare_frame,rng)
    static=base[STATIC].copy()
    static.loc[rare,STATIC]=rare_frame[STATIC].to_numpy()
    pd.testing.assert_frame_equal(static.loc[~rare],base.loc[~rare,STATIC],check_dtype=False)
    assert np.array_equal(_detector_mask(static,*detector),rare)
    assert not static[STATIC].merge(b[STATIC],how='inner').shape[0]
    static.insert(0,'record_id',np.arange(len(static)))
    return static,{'proposed':proposed,'accepted_before_truncation':count,'rare_count':n}


def diagnostics(frame,model,detector):
    mask=_detector_mask(frame,*detector);res=residual_values(frame,model)
    return {name:{'n':int(m.sum()),'FPG_residual_mean':float(res[m].mean()),
        'FPG_residual_median':float(np.median(res[m])),'onset_rate':float(frame.loc[m,'onset'].mean())}
        for name,m in [('detected_rare',mask),('common',~mask)]}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,required=True,choices=[0,1,2]);args=ap.parse_args()
    out=OUT/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed seed must not be rerun')
    b=pd.read_csv(DIST/'B.csv');base=pd.read_csv(BASE);cfg=load_cfg();detector=_load_detector(DIST)
    model=fit_residual(b,detector)
    dump(out/'residual_model.json',{k:v.tolist() for k,v in model.items()})
    dump(out/'baseline_diagnostics.json',{'B_detected_groups':diagnostics(b,model,detector),'baseline':diagnostics(base,model,detector)})
    bs=json.loads((BASE.parent/'rare_aligned_s0_score.json').read_text(encoding='utf-8'))
    ob,db=_fit_cause_cox(b,'onset'),_fit_cause_cox(b,'death')
    folds={s:Fold(s) for s in [2401,2402,2403]}
    refs={s:json.loads(next((ROOT/'research/runs/20260924-G03b-001/evaluation').glob(f'C_rare_aligned*fold{s}.json')).read_text(encoding='utf-8')) for s in folds}
    records=[]
    for alpha in [0.,.5,1.]:
        static,trace=generate(base,b,cfg,detector,model,alpha,args.seed)
        static.to_csv(out/f'static_a{alpha:.1f}.csv',index=False)
        dump(out/f'generation_a{alpha:.1f}.json',trace)
        trials=_run_path(b,static,cfg,ob,db,args.seed,1.,.5,2)
        if len(trials)!=3:
            dump(out/f'cox_incomplete_a{alpha:.1f}.json',{'expected':3,'actual':len(trials),'reason':'Cox convergence early stop; keep completed trials'})
        for trial in trials:
            start=time.monotonic();key=f'residual_s{args.seed}_a{alpha:.1f}_i{trial.iteration}'
            checkpoint=out/f'{key}_manifest.json'
            if checkpoint.exists():
                old=json.loads(checkpoint.read_text(encoding='utf-8'));assert old['candidate_sha256']==digest(ROOT/old['candidate'])
                records.append(old);continue
            csv=out/f'C_{key}.csv';archive=out/f'{key}.zip'
            write_c(trial.frame,csv)
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for name,path in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                    info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
            score=validate_score(csv,archive,out,key)
            prune=score['utility']<bs['utility']-.02
            risk={} if prune else {s:risk_gates(f.evaluate(csv,out),refs[s]) for s,f in folds.items()}
            deltas={k:score['facets'][k]-bs['facets'][k] for k in score['facets']}
            gates={'utility':score['utility']>=bs['utility']+.01,
                'other_facets':all(deltas[k]>=-.01 for k in ['U_gen','U_rare','U_valid']),
                'risk':not prune and max(v for rr in risk.values() for v in rr.values())<=.01}
            record={'key':key,'seed':args.seed,'alpha':alpha,'iteration':trial.iteration,'primary_setting':trial.iteration in [0,2],
                'candidate':csv.relative_to(ROOT).as_posix(),'candidate_sha256':digest(csv),'zip':archive.relative_to(ROOT).as_posix(),'zip_sha256':digest(archive),
                'utility':score['utility'],'facets':score['facets'],'facet_deltas':deltas,'validator':'OK','risk_deltas':risk,'gates':gates,'pass':all(gates.values()),
                'pruned':prune,'prune_reason':'U < baseline - .02' if prune else None,'diagnostics':diagnostics(trial.frame,model,detector),
                'beta_generator':trial.beta_generator.tolist(),'generation':trace,'seconds':time.monotonic()-start,
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'inputs_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [BASE,DIST/'B.csv',DIST/'utility_ref.json',DIST/'scoring_config.yaml',Path(__file__)]},
                'label':'exploratory local U and proxy risks, no final Anon or server submission'}
            dump(checkpoint,record);records.append(record)
            print(json.dumps({k:record[k] for k in ['key','utility','pruned','gates','seconds']}),flush=True)
    dump(out/'summary.json',{'seed':args.seed,'candidates':records,'n_candidates':len(records)})


if __name__=='__main__':main()
