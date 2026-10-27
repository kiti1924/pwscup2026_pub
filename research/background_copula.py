"""G17d population rank-copula synthesis from generation-only background."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.background_transport import (DIST, BASE, NUMERIC, _load_detector, _detector_mask,
    _fit_cause_cox, _run_path, load_cfg, write_c, dump, digest, validate_score,
    _format_continuous, _sample_categorical, transport, numeric_distances)
from research.background_partition import OUT as PARTITION_OUT, partition_frame
from experiments.run_aligned_synth import _nearest_correlation
import numpy as np
import pandas as pd
from scipy.stats import norm

OUT=ROOT/'research/runs/20260924-G17d-001'


def fit_model(generation, groups):
    """Input is generation-only numeric rows, never evaluation controls or labels."""
    model={}
    for group in [True,False]:
        x=generation.loc[np.asarray(groups)==group,NUMERIC]
        if len(x)<8:raise ValueError('Insufficient generation group')
        q=(x.rank(method='average').to_numpy()-.5)/len(x)
        z=norm.ppf(q)
        corr=_nearest_correlation(np.corrcoef(z,rowvar=False))
        assert np.linalg.eigvalsh(corr).min()>0
        model[str(group)]={'n':len(x),'correlation':corr.tolist()}
    return model


def load_model(b,bg,detector):
    split=partition_frame(bg,2701)
    pd.testing.assert_frame_equal(split,pd.read_csv(PARTITION_OUT/'fold2701_split.csv'),check_exact=True)
    manifest=json.loads((PARTITION_OUT/'partition_manifest.json').read_text(encoding='utf-8'))
    assert digest(DIST/'A_bg.csv')==manifest['background_sha256']
    prior=json.loads((PARTITION_OUT/'seed0/utility_summary.json').read_text(encoding='utf-8'))['candidates'][0]['inputs_sha256']
    for p in [DIST/'B.csv',DIST/'A_bg.csv',DIST/'utility_ref.json',DIST/'scoring_config.yaml']:
        assert digest(p)==prior[p.relative_to(ROOT).as_posix()]
    indices=split.loc[split.partition=='generation','background_row'].to_numpy()
    generation=bg.iloc[indices].reset_index(drop=True)
    model=fit_model(generation,_detector_mask(generation,*detector))
    return model,split.loc[split.partition=='generation'].reset_index(drop=True)


def generate(b,cfg,detector,model,shrink,seed):
    if shrink not in [0.,.15]:raise ValueError('Unregistered shrinkage')
    rng=np.random.default_rng(20000+seed)
    latent=[];raw=[];covariances={}
    for group,n in [(True,59),(False,990)]:
        corr=np.asarray(model[str(group)]['correlation'])
        cov=(1-shrink)*corr+shrink*np.eye(7)
        noise=rng.standard_normal((n,7))
        z=noise@np.linalg.cholesky(cov).T
        raw.append(pd.DataFrame(noise,columns=NUMERIC));latent.append(pd.DataFrame(z,columns=NUMERIC))
        covariances[str(group)]=cov.tolist()
    noise=pd.concat(raw,ignore_index=True);z=pd.concat(latent,ignore_index=True)
    groups=np.r_[np.ones(59,bool),np.zeros(990,bool)]
    target_groups=_detector_mask(b,*detector)
    assert len(b)==1049 and target_groups.sum()==59
    continuous,trace=transport(z,groups,b,target_groups,1.)
    static=_sample_categorical(b,_format_continuous(continuous,cfg),rng)
    static.insert(0,'record_id',np.arange(len(static)))
    return static,noise,z,trace,covariances


def save_frame(path,frame):
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path),frame,check_dtype=False,rtol=1e-13,atol=1e-13)
    else:frame.to_csv(path,index=False,lineterminator='\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,required=True,choices=[0,1,2]);args=ap.parse_args()
    out=OUT/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'utility_summary.json').exists():raise FileExistsError('Completed seed must not be rerun')
    if args.seed:
        assert json.loads((OUT/'seed0/utility_summary.json').read_text(encoding='utf-8'))['expand_seeds']
    b=pd.read_csv(DIST/'B.csv');bg=pd.read_csv(DIST/'A_bg.csv');cfg=load_cfg();detector=_load_detector(DIST)
    model,indices=load_model(b,bg,detector)
    modelpath=OUT/'generation_model.json'
    if modelpath.exists():assert json.loads(modelpath.read_text())==model
    else:dump(modelpath,model)
    save_frame(OUT/'generation_indices.csv',indices)
    ob,db=_fit_cause_cox(b,'onset'),_fit_cause_cox(b,'death')
    inputs=[DIST/'B.csv',DIST/'A_bg.csv',DIST/'utility_ref.json',DIST/'scoring_config.yaml',
        PARTITION_OUT/'fold2701_split.csv',modelpath,Path(__file__),ROOT/'research/background_transport.py',
        ROOT/'experiments/run_aligned_synth.py',ROOT/'experiments/run_calibrated_outcomes.py']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();records=[]
    for shrink in [0.,.15]:
        static,noise,z,trace,cov=generate(b,cfg,detector,model,shrink,args.seed)
        for name,frame in [('static',static),('noise',noise),('latent',z),('transport',trace)]:
            save_frame(out/f'{name}_s{shrink:.2f}.csv',frame)
        dump(out/f'covariance_s{shrink:.2f}.json',cov)
        distances=numeric_distances(static,b,bg)
        detected=int(_detector_mask(static,*detector).sum())
        trials=_run_path(b,static,cfg,ob,db,args.seed,1.,.5,6)
        if len(trials)!=7:dump(out/f'cox_incomplete_s{shrink:.2f}.json',{'expected':7,'actual':len(trials),'reason':'Cox convergence stop; no replacement'})
        for trial in trials:
            started=time.monotonic();key=f'bgcopula_s{args.seed}_r{shrink:.2f}_i{trial.iteration}';checkpoint=out/f'{key}_manifest.json'
            if checkpoint.exists():
                r=json.loads(checkpoint.read_text(encoding='utf-8'));assert r['inputs_sha256']==hashes
                assert digest(ROOT/r['candidate'])==r['candidate_sha256'];records.append(r);continue
            csv,archive=out/f'C_{key}.csv',out/f'{key}.zip';write_c(trial.frame,csv)
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as archive_file:
                for name,path in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                    info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                    archive_file.writestr(info,path.read_bytes())
            score=validate_score(csv,archive,out,key)
            r={'key':key,'seed':args.seed,'shrink':shrink,'iteration':trial.iteration,'primary_setting':trial.iteration in [2,4,6],
                'candidate':csv.relative_to(ROOT).as_posix(),'candidate_sha256':digest(csv),
                'zip':archive.relative_to(ROOT).as_posix(),'zip_sha256':digest(archive),
                'utility':score['utility'],'facets':score['facets'],'validator':'OK','pruned':score['utility']<.85,
                'privacy_state':'unmeasured','detected_rare':detected,'novel':score['detail']['U_valid']['novel'],
                'numeric_distances':distances,'beta_generator':trial.beta_generator.tolist(),'code_commit':commit,
                'inputs_sha256':hashes,'seconds':time.monotonic()-started,'label':'Exploratory local utilities, no Anon or adoption'}
            dump(checkpoint,r);records.append(r)
            print(json.dumps({k:r[k] for k in ['key','utility','facets','novel','detected_rare','pruned']}),flush=True)
    dump(out/'utility_summary.json',{'seed':args.seed,'n_candidates':len(records),'expand_seeds':any(r['primary_setting'] and r['utility']>=.85 for r in records),'candidates':records})


if __name__=='__main__':main()
