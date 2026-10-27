"""FA95 public-only generation, exact old replay, fixed-MIA package; never submits."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import yaml
import fa55_public_api as api

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pwscup2026-kit/starter/src'))
from pwscup2026.kit.validate import validate_mia_submission,validate_aia_submission,validate_submission_zip


def read(path):return pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv(path,frame):pd.DataFrame(frame).to_csv(path,index=False,lineterminator='\n')
def bits(a,b):np.testing.assert_array_equal(np.asarray(a,float).view(np.uint64),np.asarray(b,float).view(np.uint64))


def load_plan(out):
    p=json.loads((out/'plan.json').read_text())
    assert subprocess.check_output(['git','-C',str(ROOT/'pwscup2026-kit'),'rev-parse','HEAD'],text=True).strip()==p['kit_commit']
    return p


def budget(out,plan,start,stage):
    prior=sum(json.loads(p.read_text())['seconds'] for p in out.glob('*_complete.json') if p.name in ['replay_complete.json','main_complete.json','package_complete.json','verification_complete.json'])
    if (out/'failed_compute.json').exists():prior+=json.loads((out/'failed_compute.json').read_text())['seconds']
    if prior+time.monotonic()-start>plan['compute_budget_seconds']:
        raise TimeoutError('FA95 fixed cumulative compute budget exhausted; target checkpoints retained')


def audit_open(world):
    """Assert public-only inference IO. Expected output artifacts stay in FA95 run."""
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=str(args[0])
        if '/sources/truth' in path or '/sources/attacks' in path or path.endswith('/_private_team14/B.csv'):
            raise RuntimeError('Forbidden truth/teacher/private B read in inference')
        if world=='main' and ('_private_team10/' in path or 'prelim_attack_' in path or ('research/runs/' in path and 'FA95-fa55-public-api-001' not in path)):
            raise RuntimeError('Historical/preliminary artifacts forbidden during main inference')
    sys.addaudithook(hook)


def generate(out,world):
    plan=load_plan(out);start=time.monotonic();stage='replay' if world=='preliminary' else 'main'
    assert not (out/(stage+'_complete.json')).exists()
    if world=='main':assert (out/'replay_complete.json').exists()
    audit_open(world)
    for p,h in plan['public_hashes'][world].items():assert sha(ROOT/p)==h,p
    pool=plan['preliminary_pool' if world=='preliminary' else 'main_pool']
    ks=plan['preliminary_targets' if world=='preliminary' else 'main_targets']
    folder=ROOT/plan['public_root'][world]
    cs={k:read(folder/f'C_{k}.csv') for k in pool};bg=read(ROOT/plan['background'][world])
    asset=api.prepare(bg,cs,pool);dest=out/world;dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'scale.npz',mean=asset['scale']['mean'],sd=asset['scale']['sd'],pool=pool)
    fp=[];sexrows=[];checks=[]
    for k in ks:
        budget(out,plan,start,stage);target=dest/f'target{k}'
        if (target/'checkpoint.json').exists():
            for p,h in json.loads((target/'checkpoint.json').read_text()).items():assert sha(target/p)==h
            fp.extend(json.loads((target/'fingerprint.json').read_text())['rows']);sexrows.extend(json.loads((target/'fingerprint.json').read_text())['sex'])
            if world=='preliminary':checks.append(json.loads((target/'replay_check.json').read_text()))
            continue
        q=read(folder/f'aia_challenge_{k}.csv');result=api.predict(asset,q,k,set(q.challenge_row_id))
        target.mkdir();csv(target/'query.csv',q);csv(target/'predictions.csv',dict(challenge_row_id=q.challenge_row_id,time_hat=result['prediction']))
        np.savez_compressed(target/'evidence.npz',**{n:v for n,v in result.items() if n!='donor_public_rows'})
        csv(target/'donor_public_rows.csv',pd.DataFrame(result['donor_public_rows'],columns=api.PUBLIC))
        # Replay comparisons are isolated to preliminary stage; main never opens these.
        if world=='preliminary':
            source=ROOT/f'research/runs/20261010-FA55-selection-ensemble-001/target{k}'
            for path in [source/'inputs.npz',source/'selection.npz',source/'predictions.csv',ROOT/f'research/runs/20261010-FA50-distance-norms-001/aia/target{k}/l1_log5_distances.npz']:
                assert sha(path)==plan['reference_hashes'][str(path.relative_to(ROOT))]
            with np.load(source/'inputs.npz') as z:inputs=dict(z)
            with np.load(source/'selection.npz') as z:selection=dict(z)
            with np.load(ROOT/f'research/runs/20261010-FA50-distance-norms-001/aia/target{k}/l1_log5_distances.npz') as z:old=dict(z)
            saved=read(source/'predictions.csv').set_index('challenge_row_id').loc[q.challenge_row_id]
            comparison=dict(target=k,rows=len(q))
            for n in ['v1','v2']:
                diff=result[n]-inputs[n]
                comparison[n+'_max_abs']=float(np.max(abs(diff)));comparison[n+'_mean_abs']=float(np.mean(abs(diff)))
                np.testing.assert_allclose(result[n],inputs[n],atol=plan['evidence_atol'],rtol=plan['evidence_rtol'])
            np.testing.assert_array_equal(result['mask'],selection['mean']);bits(result['donor_time'],inputs['time']);bits(result['prediction'],saved['mean'])
            olddonor=cs[k].iloc[old[f'C{k}_index']].reset_index(drop=True)[api.PUBLIC]
            pd.testing.assert_frame_equal(read(target/'donor_public_rows.csv'),olddonor,check_dtype=False,check_exact=True)
            comparison.update(mask_exact=True,donor_time_exact=True,donor_row_exact=True,prediction_exact=True)
            dump(target/'replay_check.json',comparison);checks.append(comparison)
        row,sex=api.fingerprint(asset,q,k,result)
        rows=[dict(world=world,target=k,preliminary_gain4=world=='preliminary' and k in [1,13,22,24],**row)]
        sx=[dict(world=world,target=k,**s) for s in sex]
        fp.extend(rows);sexrows.extend(sx);dump(target/'fingerprint.json',dict(rows=rows,sex=sx))
        dump(target/'checkpoint.json',{p.name:sha(p) for p in sorted(target.iterdir())})
        print('FA95',world,'checkpoint',k,flush=True)
    csv(dest/'fingerprints.csv',fp);csv(dest/'selection_by_sex.csv',sexrows)
    if checks:csv(out/'preliminary_replay.csv',checks)
    dump(out/(stage+'_complete.json'),dict(seconds=time.monotonic()-start,targets=len(ks),rows=len(ks)*326,
         code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),world=world,
         truth_read=False,teacher_read=False,private_B_read=False,public_self_in_rival_pool=True,
         new_scores=0,new_submissions=0))


def assemble_zip(folder,mia,aia,token):
    folder.mkdir(exist_ok=False)
    (folder/'F_mia.csv').write_bytes(mia);(folder/'F_aia.csv').write_bytes(aia);(folder/'token.txt').write_bytes(token)
    with zipfile.ZipFile(folder/'candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n,data in [('F_mia.csv',mia),('F_aia.csv',aia),('token.txt',token)]:
            info=zipfile.ZipInfo(n,(2026,10,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)


def package(out):
    start=time.monotonic();plan=load_plan(out);assert (out/'main_complete.json').exists()
    for p,h in plan['fixed_hashes'].items():assert sha(ROOT/p)==h,p
    frames=[]
    for k in plan['main_targets']:
        p=read(out/f'main/target{k}/predictions.csv');p.insert(0,'target_k',k);frames.append(p)
    aia=pd.concat(frames,ignore_index=True).sort_values(['target_k','challenge_row_id']).reset_index(drop=True)
    assert len(aia)==7172 and not aia.duplicated(['target_k','challenge_row_id']).any()
    with zipfile.ZipFile(ROOT/plan['source_zips']['972302']) as z:mia=z.read('F_mia.csv');token=z.read('token.txt')
    assert token==(ROOT/'_private_team14/token.txt').read_bytes()
    assemble_zip(out/'candidate',mia,aia.to_csv(index=False,lineterminator='\n').encode(),token)
    validation=validators(out/'candidate/candidate.zip',plan)
    differences(out,plan,aia)
    fingerprint_comparison(out,plan)
    budget(out,plan,start,'package')
    dump(out/'candidate/manifest.json',dict(method=api.VERSION,candidate='FA55 mean',measured=False,score=None,
        expected_gain=None,mia_source='972302',mia_source_bytes_exact=True,targets=plan['main_targets'],
        rival_pool=plan['main_pool'],rival_exclusion='only target itself',zip_sha256=sha(out/'candidate/candidate.zip'),
        aia_rows=7172,mia_rows=1049,official_validators=validation,new_submissions=0))
    dump(out/'package_complete.json',dict(seconds=time.monotonic()-start,zip_sha256=sha(out/'candidate/candidate.zip'),
        validators=validation,new_scores=0,new_submissions=0))
    print('FA95 packaged unmeasured candidate, 3 validators pass',flush=True)


def validators(path,plan):
    cfg=yaml.safe_load((ROOT/'_private_team14/scoring_config.yaml').read_text())
    dist=ROOT/plan['public_root']['main'];targets=plan['main_targets']
    # Private B is only read for validator key membership; never passed to inference.
    ids=set(read(ROOT/'_private_team14/B.csv').record_id)
    strict={'team_id':14,'mia_columns':{'target_columns':[str(k) for k in plan['main_pool']]},'attacker_record_ids':ids,
            'challenge':{k:set(read(dist/f'aia_challenge_{k}.csv').challenge_row_id) for k in targets}}
    with zipfile.ZipFile(path) as z:
        m=read(io.BytesIO(z.read('F_mia.csv')));a=read(io.BytesIO(z.read('F_aia.csv')))
    assert len(m)==1049 and len(a)==7172 and sorted(a.target_k.unique())==targets
    assert set(m.record_id)==ids
    result=[]
    for name,value in [('MIA',validate_mia_submission(m,cfg,strict)),('AIA',validate_aia_submission(a,cfg,strict)),
                       ('ZIP',validate_submission_zip(path,cfg,dist,'attack'))]:
        assert value.ok,'Official '+name+' validator failed (token contents suppressed)'
        result.append(dict(validator=name,passed=True))
    return result


def differences(out,plan,aia):
    refs={}
    for sid,path in plan['source_zips'].items():
        with zipfile.ZipFile(ROOT/path) as z:ref=read(io.BytesIO(z.read('F_aia.csv')))
        assert not ref.duplicated(['target_k','challenge_row_id']).any()
        refs[sid]=ref.set_index(['target_k','challenge_row_id']).loc[pd.MultiIndex.from_frame(aia[['target_k','challenge_row_id']])].time_hat.to_numpy()
    rows=[];all_new=np.ones(len(aia),bool)
    for sid,v in refs.items():
        changed=aia.time_hat.to_numpy().view(np.uint64)!=v.view(np.uint64)
        all_new &= changed
        for k in plan['main_targets']:
            ix=aia.target_k.eq(k).to_numpy();rows.append(dict(target=k,source=sid,rows=int(ix.sum()),
                 changed_rows=int(changed[ix].sum()),identical_vector=not changed[ix].any(),
                 candidate_unique_times=int(aia.time_hat[ix].nunique()),source_unique_times=int(len(np.unique(v[ix]))),
                 candidate_near95=int((abs(aia.time_hat.to_numpy()[ix]-9.5)<=.5).sum()),source_near95=int((abs(v[ix]-9.5)<=.5).sum())))
    csv(out/'main_answer_differences.csv',rows)
    csv(out/'main_new_value_rows.csv',dict(target_k=aia.target_k,challenge_row_id=aia.challenge_row_id,
        candidate_time=aia.time_hat,new_vs_all4=all_new,**{sid+'_time':v for sid,v in refs.items()}))
    csv(out/'main_new_value_by_target.csv',[dict(target=k,new_vs_all4=int(all_new[aia.target_k.eq(k)].sum())) for k in plan['main_targets']])
    frequency=aia.groupby(['target_k','time_hat']).size().reset_index(name='count');csv(out/'main_time_distribution.csv',frequency)


def fingerprint_comparison(out,plan):
    pre=read(out/'preliminary/fingerprints.csv');main=read(out/'main/fingerprints.csv')
    gain=pre[pre.target.isin([1,13,22,24])];rows=[]
    for feature in plan['fingerprint_features']:
        lo=float(gain[feature].min());hi=float(gain[feature].max())
        for r in main.itertuples():
            value=float(getattr(r,feature))
            rows.append(dict(target=r.target,feature=feature,value=value,gain4_min=lo,gain4_max=hi,
                gain4_median=float(gain[feature].median()),pre20_median=float(pre[feature].median()),
                pre20_min=float(pre[feature].min()),pre20_max=float(pre[feature].max()),
                within_gain4_range=lo<=value<=hi,descriptive_only=True))
    csv(out/'fingerprint_gain4_main.csv',rows)
    csv(out/'fingerprint_group_summary.csv',[dict(group=group,feature=n,n_targets=len(df),minimum=float(df[n].min()),
        median=float(df[n].median()),maximum=float(df[n].max())) for group,df in [('pre_gain4',gain),('pre20',pre),('main22',main)] for n in plan['fingerprint_features']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['replay','main','package'],required=True)
    a=p.parse_args()
    if a.stage=='package':package(a.out)
    else:generate(a.out,'preliminary' if a.stage=='replay' else 'main')
