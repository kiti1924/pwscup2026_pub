"""G18 controlled lawful CSV edits for requested local/server metric verification."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.residual_rare import (DIST, BASE, _load_detector, _detector_mask, load_cfg,
    write_c, dump, digest, validate_score)
from pwscup2026.scoring.utility import _c2st_raw, _canon_rows, _mass_star
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

OUT = ROOT / 'research/runs/20260924-G18-001'
HIGH = ROOT / 'research/runs/20260924-G05b-001/seed0/C_bounded_s0_b_group_i2.csv'
COX = ROOT / 'research/runs/20260924-G02b-001/seed0/C_u0_h1.000_lr0.500_i2.csv'
NUM = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']


def change_mass(frame, target, detector, cfg):
    mu, inv, threshold = detector
    mask = _detector_mask(frame, *detector); current = int(mask.sum())
    want_rare = target > current; need = abs(target-current)
    x = frame[NUM].to_numpy(float)-mu
    a = inv[-1, -1]; choices = []
    lo, hi = cfg['schema']['ranges']['FPG']
    for i in np.flatnonzero(mask != want_rare):
        lin = 2 * (x[i, :-1] @ inv[:-1, -1])
        constant = x[i, :-1] @ inv[:-1, :-1] @ x[i, :-1] - threshold**2
        disc = lin**2 - 4*a*constant
        if disc <= 0: continue
        low, high = (-lin-np.sqrt(disc))/(2*a)+mu[-1], (-lin+np.sqrt(disc))/(2*a)+mu[-1]
        options = [low-1e-5, high+1e-5] if want_rare else [low+1e-5, high-1e-5]
        options = [v for v in options if lo <= v <= hi]
        if options:
            after = min(options, key=lambda v: abs(v-frame.FPG.iloc[i]))
            choices.append((abs(after-frame.FPG.iloc[i]), i, after))
    choices.sort()
    if len(choices) < need: raise ValueError('Not enough feasible one-column crossings')
    c = frame.copy(); trace = []
    for distance, i, after in choices[:need]:
        trace.append({'row': int(i), 'before_FPG': float(c.FPG.iloc[i]), 'after_FPG': float(after), 'change': float(distance)})
        c.loc[i, 'FPG'] = after
    assert int(_detector_mask(c, *detector).sum()) == target
    pd.testing.assert_frame_equal(c.drop(columns='FPG'), frame.drop(columns='FPG'))
    return c, trace


def diagnostic_c2st(c, truth, markers, gate, trees, seed):
    X = np.vstack([_canon_rows(c, markers), _canon_rows(truth, markers)])
    y = np.r_[np.ones(len(c)), np.zeros(len(truth))]
    cv = StratifiedKFold(n_splits=min(5, len(c), len(truth)), shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=1)
    p = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return float(np.clip(1-2*abs(roc_auc_score(y,p)-.5),0,1))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'prepared.json').exists(): raise FileExistsError('Prepared checkpoint exists; use --stage score')
    cfg = load_cfg(); detector = _load_detector(DIST)
    base = pd.read_csv(BASE); high = pd.read_csv(HIGH); bs = pd.read_csv(DIST/'B_self.csv')
    truth = bs.loc[bs.is_rare.astype(bool)]
    rare_b = bs.loc[_detector_mask(bs, *detector)]
    ucfg = cfg['scoring']['u_rare']; markers = ucfg['markers'] or NUM
    gate = max(ucfg['min_rare_n'], ucfg['c2st']['min_rare_n']); trees = ucfg['c2st']['n_estimators']
    candidates = {}; traces = {}
    exact = base.copy(); exact.iloc[:20] = np.repeat(base.iloc[[0]].to_numpy(),20,axis=0)
    tiny = exact.copy(); tiny.loc[:19,'TG'] += np.arange(20)*1e-10
    assert exact.duplicated().sum() == 19 and tiny.duplicated().sum() == 0
    candidates['dup_exact'] = exact; candidates['dup_epsilon'] = tiny
    traces['dup'] = {'duplicated_rows': 19, 'edited_rows': list(range(20)), 'TG_max_delta': float(np.max(abs(tiny.TG-exact.TG)))}
    for n in [73, 100]:
        candidates[f'mass{n}'], traces[f'mass{n}'] = change_mass(base,n,detector,cfg)
    slack = base.copy(); cens = (slack.onset==0)&(slack.death==0)
    assert np.all(slack.loc[cens,'time']==10)
    slack.loc[cens,'time'] = 9.999; candidates['censor_slack'] = slack
    traces['censor_slack'] = {'n_changed': int(cens.sum()), 'columns': ['time'], 'from': 10, 'to': 9.999}
    parent_mask = _detector_mask(high,*detector); indices = np.flatnonzero(parent_mask)
    raw_base = _c2st_raw(rare_b,truth,markers,gate,trees)
    raw_parent = _c2st_raw(high.loc[parent_mask],truth,markers,gate,trees)
    pd.testing.assert_frame_equal(high, pd.read_csv(HIGH))
    scan = []; folder = OUT / 'c2st_scan'; folder.mkdir(exist_ok=True)
    for seed in range(18000,18032):
        path = folder / f'C_swap{seed}.csv'; record = folder / f'swap{seed}.json'
        if record.exists():
            r=json.loads(record.read_text(encoding='utf-8')); assert digest(path)==r['candidate_sha256']; scan.append(r); continue
        rng=np.random.default_rng(seed); selected=rng.choice(indices,8,replace=False)
        c=high.copy(); original=c.FPG.to_numpy().copy(); pairs=selected.reshape(4,2)
        for i,j in pairs: c.loc[[i,j],'FPG']=[original[j],original[i]]
        np.testing.assert_array_equal(np.sort(c.FPG),np.sort(high.FPG))
        c.to_csv(path,index=False); c=pd.read_csv(path)
        mask=_detector_mask(c,*detector); raw=_c2st_raw(c.loc[mask],truth,markers,gate,trees)
        r={'seed':seed,'candidate':path.relative_to(ROOT).as_posix(),'candidate_sha256':digest(path),
           'pairs':pairs.tolist(),'rare_detected':int(mask.sum()),'raw':raw,'base_raw':raw_base,'calibrated':float(min(1,raw/raw_base))}
        dump(record,r);scan.append(r)
        print(json.dumps({'scan_seed':seed,'c2st_cal':r['calibrated']}),flush=True)
    best=max(scan,key=lambda r:(r['calibrated'],r['raw'],-r['seed']))
    improved=best['raw']>raw_parent
    if not improved: best=max(scan,key=lambda r:(abs(r['raw']-raw_parent),-r['seed']))
    winner=pd.read_csv(ROOT/best['candidate'])
    candidates['c2st_control']=high; candidates['c2st_selected']=winner
    candidates['cox_calibrated']=pd.read_csv(COX)
    alternate=[]
    for seed in [0,1,2,3]:
        ref=diagnostic_c2st(rare_b,truth,markers,gate,trees,seed)
        for name,c in [('control',high),('selected',winner)]:
            group=c.loc[_detector_mask(c,*detector)]
            value=diagnostic_c2st(group,truth,markers,gate,trees,seed)
            if seed==0: np.testing.assert_allclose(value,_c2st_raw(group,truth,markers,gate,trees),atol=1e-15)
            alternate.append({'candidate':name,'diagnostic_seed':seed,'raw':value,'base_raw':ref,'calibrated':min(1,value/ref)})
    shuffled=winner.sample(frac=1,random_state=18099)
    np.testing.assert_allclose(_c2st_raw(shuffled.loc[_detector_mask(shuffled,*detector)],truth,markers,gate,trees),best['raw'],atol=1e-15)
    pd.DataFrame(alternate).to_csv(OUT/'c2st_other_seeds.csv',index=False)
    dump(OUT/'c2st_selection.json',{'selected':best,'control_raw':raw_parent,'control_calibrated':min(1,raw_parent/raw_base),
        'raw_improved':improved,'calibrated_improved':best['calibrated']>min(1,raw_parent/raw_base),
        'whole_U_improved':None,'selection':'maximum official calibrated/raw; if no gain select largest difference',
        'row_order_invariant':True,'limitations':'Alternate seed diagnostics occur after selection; no server internal seed inference'})
    for key,c in candidates.items():
        assert list(c.columns)==list(base.columns) and c.shape==base.shape
        assert np.isfinite(c.drop(columns='prefecture').to_numpy(float)).all()
        assert c.prefecture.notna().all()
        write_c(c,OUT/f'C_{key}.csv')
    dump(OUT/'edit_traces.json',traces)
    dump(OUT/'prepared.json',{'candidates':list(candidates),'n':len(candidates),
        'inputs_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [BASE,HIGH,COX,DIST/'B.csv',DIST/'B_self.csv',DIST/'utility_ref.json',DIST/'scoring_config.yaml',Path(__file__)]},
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'diagnostic_only':True,'incumbent_submission':942802})


def amend():
    if (OUT/'prepared_additional.json').exists(): raise FileExistsError('Amendment already prepared')
    base=pd.read_csv(BASE);mask=(base.onset==0)&(base.death==0)
    c=base.copy();c.loc[mask,'time']=9.9
    write_c(c,OUT/'C_censor_outside.csv')
    source=OUT/'c2st_scan/C_swap18019.csv'
    write_c(pd.read_csv(source),OUT/'C_c2st_low.csv')
    dump(OUT/'prepared_additional.json',{'candidates':['censor_outside','c2st_low'],
         'reason':'Original censor shift stayed inside atol .01; original C2ST control and selected calibrated scores both saturated',
         'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [BASE,source]},
         'submission_exclusions':['censor_slack','c2st_control'],'additional_search':False})


def score():
    prepared=json.loads((OUT/'prepared.json').read_text(encoding='utf-8')); records=[]
    keys=prepared['candidates']
    if (OUT/'prepared_additional.json').exists():
        keys=keys+json.loads((OUT/'prepared_additional.json').read_text(encoding='utf-8'))['candidates']
    for key in keys:
        csv=OUT/f'C_{key}.csv';archive=OUT/f'{key}.zip';manifest=OUT/f'{key}_manifest.json'
        if manifest.exists():
            r=json.loads(manifest.read_text(encoding='utf-8'));assert r['candidate_sha256']==digest(csv);records.append(r);continue
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,path in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
        result=validate_score(csv,archive,OUT,key)
        r={'key':key,'candidate':csv.relative_to(ROOT).as_posix(),'candidate_sha256':digest(csv),
            'zip':archive.relative_to(ROOT).as_posix(),'zip_sha256':digest(archive),
            'utility':result['utility'],'facets':result['facets'],'validator':'OK','status':'local_validated_not_submitted',
            'diagnostic_only':True,'score':(OUT/f'{key}_score.json').relative_to(ROOT).as_posix()}
        dump(manifest,r);records.append(r)
        print(json.dumps({'key':key,'utility':r['utility'],'facets':r['facets']}),flush=True)
    dump(OUT/'local_summary.json',{'candidates':records,'n':len(records),'server_verified':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['prepare','score','amend'],required=True)
    args=ap.parse_args();{'prepare':prepare,'score':score,'amend':amend}[args.stage]()
