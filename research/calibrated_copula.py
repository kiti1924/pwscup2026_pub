"""G17e: aggregate correlation and detector-group hazard calibration."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.background_copula import (DIST,BASE,PARTITION_OUT,fit_model,load_model,generate,save_frame,
    _load_detector,_detector_mask,_fit_cause_cox,load_cfg,write_c,dump,digest,validate_score,numeric_distances,
    _nearest_correlation)
from experiments.run_calibrated_outcomes import (_COX_COLUMNS,_cox_frame,_center_vector,_fixed_uniforms,
    _sample_event_time,_TIME_FLOOR_YEARS,coerce_to_submission,_score_u_spec,_beta_from_detail,_simulate_outcomes)
import numpy as np
import pandas as pd
from scipy.special import logit

OUT=ROOT/'research/runs/20260924-G17e-001'


def blend_models(g_model,b_model,weight):
    if weight not in [.5,1.]:raise ValueError('Unregistered weight')
    return {group:{'correlation':_nearest_correlation((1-weight)*np.asarray(g_model[group]['correlation'])+
              weight*np.asarray(b_model[group]['correlation'])).tolist(),
              'n_G':g_model[group]['n'],'n_B':b_model[group]['n']} for group in ['True','False']}


def rates(frame,mask):
    values=[]
    for group in [False,True]:
        part=np.asarray(mask)==group;n=int(part.sum())
        if n==0:raise ValueError('Empty detector group')
        count=int(frame.loc[part,'onset'].sum())
        values.append({'n':n,'events':count,'rate':count/n,'smoothed':(count+.5)/(n+1)})
    return values


def update_gamma(gamma,target,current):
    return np.clip(np.asarray(gamma)+.5*(logit([x['smoothed'] for x in target])-logit([x['smoothed'] for x in current])),-2.,2.)


def simulate(static,cfg,onset_bundle,death_bundle,beta,gamma,mask,u,v):
    om,ot,oh=onset_bundle;dm,dt,dh=death_bundle
    if np.all(np.asarray(gamma)==0):
        return _simulate_outcomes(static,cfg,om,ot,oh,dm,dt,dh,beta,1.,u,v)
    x=_cox_frame(static);centered=x[_COX_COLUMNS].to_numpy(float)-_center_vector(om)
    logph=np.clip(centered@beta+np.asarray(gamma)[np.asarray(mask,int)],-40.,40.)
    onset_time=_sample_event_time(np.exp(logph),ot,oh,u)
    death_time=_sample_event_time(dm.predict_partial_hazard(x).to_numpy(float),dt,dh,v)
    horizon=float(cfg['onset']['horizon_years']);censored=np.minimum(onset_time,death_time)>horizon
    onset_event=(~censored)&(onset_time<=death_time);death_event=(~censored)&(death_time<onset_time)
    frame=static.copy(deep=True);times=np.full(len(static),horizon)
    times[onset_event]=onset_time[onset_event];times[death_event]=death_time[death_event]
    frame['time']=np.clip(times,_TIME_FLOOR_YEARS,horizon)
    frame['onset']=onset_event.astype(int);frame['death']=death_event.astype(int)
    return coerce_to_submission(frame,cfg)


def run_path(b,static,cfg,ob,db,detector,seed,calibrate,iterations=6):
    target_beta=ob[0].params_.reindex(_COX_COLUMNS).to_numpy(float);beta=target_beta.copy();gamma=np.zeros(2)
    mask=_detector_mask(static,*detector);target_rates=rates(b,_detector_mask(b,*detector))
    u,v=_fixed_uniforms(seed,len(static));trials=[]
    for iteration in range(iterations+1):
        frame=simulate(static,cfg,ob,db,beta,gamma,mask,u,v)
        spec,detail=_score_u_spec(frame,b)
        if not detail.get('cox_converged',False):break
        observed=rates(frame,mask)
        trials.append({'iteration':iteration,'frame':frame,'beta_generator':beta.tolist(),'gamma':gamma.tolist(),
                       'target_rates':target_rates,'observed_rates':observed,'spec':float(spec)})
        beta=beta+.5*(target_beta-_beta_from_detail(detail));span=np.maximum(3*np.abs(target_beta),.15)
        beta=np.clip(beta,target_beta-span,target_beta+span)
        if calibrate:gamma=update_gamma(gamma,target_rates,observed)
    return trials


def setup():
    b=pd.read_csv(DIST/'B.csv');bg=pd.read_csv(DIST/'A_bg.csv');cfg=load_cfg();detector=_load_detector(DIST)
    gm,indices=load_model(b,bg,detector);bm=fit_model(b,_detector_mask(b,*detector))
    return b,bg,cfg,detector,gm,bm,indices


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=[0,1,2]);args=ap.parse_args()
    folder=OUT/f'seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'utility_summary.json').exists():raise FileExistsError('Completed seed exists')
    if args.seed:assert json.loads((OUT/'seed0/utility_summary.json').read_text())['expand_seeds']
    b,bg,cfg,detector,gm,bm,indices=setup();save_frame(OUT/'generation_indices.csv',indices)
    modelpath=OUT/'models.json';models={'generation':gm,'B_aggregate':bm}
    if modelpath.exists():assert json.loads(modelpath.read_text())==models
    else:dump(modelpath,models)
    ob,db=_fit_cause_cox(b,'onset'),_fit_cause_cox(b,'death')
    inputs=[DIST/'B.csv',DIST/'A_bg.csv',DIST/'utility_ref.json',DIST/'scoring_config.yaml',modelpath,
        PARTITION_OUT/'fold2701_split.csv',Path(__file__),ROOT/'research/background_copula.py',
        ROOT/'research/background_transport.py',ROOT/'experiments/run_aligned_synth.py',ROOT/'experiments/run_calibrated_outcomes.py']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in inputs};commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=[]
    for weight in [.5,1.]:
        mixed=blend_models(gm,bm,weight)
        static,noise,z,trace,cov=generate(b,cfg,detector,mixed,0.,args.seed)
        for name,frame in [('static',static),('noise',noise),('latent',z),('transport',trace)]:save_frame(folder/f'{name}_w{weight:.1f}.csv',frame)
        dump(folder/f'correlation_w{weight:.1f}.json',{'mixed_model':mixed,'covariance':cov})
        detected=int(_detector_mask(static,*detector).sum());distances=numeric_distances(static,b,bg)
        for calibrated in [False,True]:
            trials=run_path(b,static,cfg,ob,db,detector,args.seed,calibrated)
            if len(trials)!=7:dump(folder/f'incomplete_w{weight:.1f}_g{int(calibrated)}.json',{'expected':7,'actual':len(trials),'reason':'Cox convergence stop'})
            for trial in trials:
                started=time.monotonic();i=trial['iteration'];key=f'calcopula_s{args.seed}_w{weight:.1f}_g{int(calibrated)}_i{i}'
                checkpoint=folder/f'{key}_manifest.json'
                if checkpoint.exists():
                    r=json.loads(checkpoint.read_text());assert r['inputs_sha256']==hashes and digest(ROOT/r['candidate'])==r['candidate_sha256']
                    records.append(r);continue
                csv,archive=folder/f'C_{key}.csv',folder/f'{key}.zip';write_c(trial['frame'],csv)
                with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zfile:
                    for name,path in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                        info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                        zfile.writestr(info,path.read_bytes())
                score=validate_score(csv,archive,folder,key)
                r={'key':key,'seed':args.seed,'weight':weight,'group_calibration':calibrated,'iteration':i,'primary_setting':i in [2,4,6],
                    'candidate':csv.relative_to(ROOT).as_posix(),'candidate_sha256':digest(csv),'zip':archive.relative_to(ROOT).as_posix(),'zip_sha256':digest(archive),
                    'utility':score['utility'],'facets':score['facets'],'validator':'OK','pruned':score['utility']<.85,'privacy_state':'unmeasured',
                    'detected_rare':detected,'novel':score['detail']['U_valid']['novel'],'numeric_distances':distances,
                    'beta_generator':trial['beta_generator'],'gamma':trial['gamma'],'target_rates':trial['target_rates'],'observed_rates':trial['observed_rates'],
                    'code_commit':commit,'inputs_sha256':hashes,'seconds':time.monotonic()-started,'label':'Exploratory local utility; no final Anon or adoption'}
                dump(checkpoint,r);records.append(r)
                print(json.dumps({k:r[k] for k in ['key','utility','facets','novel','gamma','pruned']}),flush=True)
    dump(folder/'utility_summary.json',{'seed':args.seed,'n_candidates':len(records),
        'expand_seeds':any(r['primary_setting'] and r['utility']>=.85 for r in records),'candidates':records})


if __name__=='__main__':main()
