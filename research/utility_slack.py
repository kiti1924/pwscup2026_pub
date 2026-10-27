"""G22 bounded event-time jitter; inference/truth never enters generation."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from research.adaptive_privacy import dump,digest
from research.repair_sex_correlation import validate_score,DIST
from research.overnight_refinement import versioned_matrix,verify_reuse
from research.geography_free_attacks import evaluate as evaluate_geo
from research.summarize_tradeoff import METRICS
import research.background_partition as bp
OUT=ROOT/'research/runs/20260925-G22-001'
BASE=ROOT/'research/runs/20260925-G21-002/C_sex_group_i2.csv'
BASE_U=.9295762066619602
WIDTHS=[.1,.25,.5,.75,1.,1.5]

def transform(base,width):
    frame=base.copy()
    mask=((base.onset==1)|(base.death==1))&(base.time<10)
    noise=np.random.default_rng(3401).uniform(-1,1,len(base))
    low,high=.001,9.999
    raw=base.time.to_numpy()+width*noise
    phase=(raw-low)%(2*(high-low))
    reflected=low+np.where(phase<=high-low,phase,2*(high-low)-phase)
    frame.loc[mask,'time']=np.round(reflected[mask],6)
    pd.testing.assert_frame_equal(frame.drop(columns='time'),base.drop(columns='time'),check_exact=True)
    np.testing.assert_array_equal(frame.loc[~mask,'time'],base.loc[~mask,'time'])
    return frame

def generate():
    assert digest(BASE)=='e504704055056047a2c0f551f4717a74a657a69628efe56627541be1d91c3be9'
    base=pd.read_csv(BASE);records=[]
    for width in WIDTHS:
        key=f'jitter_w{width:g}_s3401';mp=OUT/f'{key}_manifest.json'
        if mp.exists():
            r=json.loads(mp.read_text());assert digest(ROOT/r['candidate'])==r['candidate_sha256'];records.append(r);continue
        frame=transform(base,width);pd.testing.assert_frame_equal(frame,transform(base,width),check_exact=True)
        csv=OUT/f'C_{key}.csv';archive=OUT/f'{key}.zip'
        if csv.exists():pd.testing.assert_frame_equal(pd.read_csv(csv),frame)
        else:frame.to_csv(csv,index=False,lineterminator='\n')
        pd.testing.assert_frame_equal(pd.read_csv(csv),frame)
        with zipfile.ZipFile(archive,'w') as z:
            for name,path in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                info=zipfile.ZipInfo(name,date_time=(2026,9,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
        with zipfile.ZipFile(archive) as z:
            assert z.read('C.csv')==csv.read_bytes() and z.read('token.txt')==(DIST/'token.txt').read_bytes()
        score=validate_score(csv,archive,OUT,key)
        r=dict(key=key,width=width,seed=3401,candidate=csv.relative_to(ROOT).as_posix(),candidate_sha256=digest(csv),
               zip=archive.relative_to(ROOT).as_posix(),zip_sha256=digest(archive),utility=score['utility'],facets=score['facets'],
               eligible=score['utility']>=BASE_U-1e-12 and score['facets']['U_gen']>=BASE_U-1e-12,
               validator='OK',reproduction='Exact deterministic transform; static/outcome labels and censor times preserved; ZIP checked',
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [BASE,Path(__file__),DIST/'utility_ref.json',DIST/'scoring_config.yaml']})
        dump(mp,r);records.append(r);print(json.dumps({k:r[k] for k in ['width','utility','facets','eligible']}),flush=True)
    pd.DataFrame([dict(width=r['width'],candidate=r['candidate'],U=r['utility'],eligible=r['eligible'],**r['facets']) for r in records]).to_csv(OUT/'utility_summary.csv',index=False,lineterminator='\n')

def evaluate(seed):
    assert seed in [2701,2702,2703]
    fold=bp.SeparatedFold(seed);results=[]
    for name in ['evaluation','geography']:(OUT/name).mkdir(exist_ok=True)
    for mp in sorted(OUT.glob('jitter_*_manifest.json')):
        r=json.loads(mp.read_text())
        if not r['eligible']:continue
        path=ROOT/r['candidate'];assert digest(path)==r['candidate_sha256']
        results += [fold.evaluate(path,OUT/'evaluation'),evaluate_geo(fold,path,OUT/'geography')]
        versioned_matrix(results).to_csv(OUT/f'fold{seed}_partial.csv',index=False,lineterminator='\n')
    old=pd.read_csv(BASE.parent/'all_methods.csv')
    reused=old[(old.candidate.eq(BASE.relative_to(ROOT).as_posix())|old.candidate.str.contains('positive_|negative_'))&old.fold.eq(seed)]
    assert reused.candidate.nunique()==4
    checked=verify_reuse(reused,fold)
    combined=pd.concat([reused,versioned_matrix(results)],ignore_index=True)
    assert not combined.duplicated(['candidate','fold','version','channel','method','group']).any()
    combined.to_csv(OUT/f'fold{seed}_matrix.csv',index=False,lineterminator='\n')
    dump(OUT/f'fold{seed}_checkpoint.json',dict(fold=seed,complete=True,split_sha256=fold.split_hash,checked_sources=checked))
    print(f'fold {seed} complete',flush=True)

def summarize():
    m=pd.concat([pd.read_csv(OUT/f'fold{s}_matrix.csv') for s in [2701,2702,2703]],ignore_index=True)
    m.to_csv(OUT/'all_methods.csv',index=False,lineterminator='\n');rows=[]
    for candidate,g in m.groupby('candidate'):
        path=ROOT/candidate;sp=path.with_name(path.stem.removeprefix('C_')+'_score.json')
        row={'candidate':candidate,'U':json.loads(sp.read_text())['utility'] if sp.exists() else None}
        for channel,group,metric,label in METRICS:
            v=g[g.channel.eq(channel)&g.group.eq(group)].groupby('fold')[metric].max()
            row[label+'_max']=v.max();row[label+'_mean']=v.mean()
            for seed,val in v.items():row[label+f'_fold{seed}']=val
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'summary.csv',index=False,lineterminator='\n')
    print(table[['candidate','U']+[x[3]+'_max' for x in METRICS]].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['generate','evaluate','summarize'],required=True);ap.add_argument('--fold',type=int);a=ap.parse_args()
    if a.stage=='generate':generate()
    elif a.stage=='evaluate':evaluate(a.fold)
    else:summarize()
