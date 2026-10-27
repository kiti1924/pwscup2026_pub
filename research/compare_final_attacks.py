"""FA02: lawful predictions first, retrospective/shadow scoring second; no submission."""
from __future__ import annotations
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pwscup2026-kit/starter/src')]
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from research.final_attack_api import (COLUMNS, STATIC, QI, NUM, VERSION, MIA_METHODS,
    AIA_METHODS, predict_mia, predict_aia, keyed_values, log_transform, block)
from research.build_final_attack_baseline import sha, read, write_json
from research.audit_attack_fingerprints import observable_features
from pwscup2026.scoring.attack_score import score_mia, score_aia

OUT = ROOT / 'research/runs/20261006-FA02-001'
PUBLIC = ROOT / 'attack_phase/data/public/prelim_attack_20260825'
TRUTH = ROOT / 'analysis/prelim_review_20260924/sources/truth'
MECHANISMS = ['raw', 'noise', 'cluster', 'pure_bg']


def config():
    cfg = yaml.safe_load((ROOT / '_private_team10/scoring_config.yaml').read_text())
    cfg['aia'].update(qi_continuous=QI[:5], block_keys=['sex', 'prefecture'], k_nn_onset=5)
    assert cfg['scoring']['low_fpr'] == .01 and cfg['scoring']['aia_rare_se_z'] == 1.64
    assert cfg['aia']['tau_time'] == .5
    return cfg


def clean_json(obj):
    if isinstance(obj, dict): return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj): return None
    if isinstance(obj, np.generic): return obj.item()
    return obj


def dump(path, obj): write_json(path, clean_json(obj))


def score_predictions(m, a, mt, at, cfg, target, group, seed=None, mechanism=None):
    # Truth is joined by keys only after predictions have been materialized.
    assert mt.record_id.is_unique and at.challenge_row_id.is_unique
    assert set(m.record_id) == set(mt.record_id)
    assert set(a.challenge_row_id) == set(at.challenge_row_id)
    mm = mt.set_index('record_id').loc[m.record_id]
    aa = at.set_index('challenge_row_id').loc[a.challenge_row_id]
    member = aa.label.eq('member').to_numpy()
    rare = aa.is_rare.to_numpy(bool)
    rows = []
    for method in MIA_METHODS:
        r = score_mia(m[method].to_numpy(), mm.member.to_numpy(int), cfg,
                      rare_mask=mm.is_rare.to_numpy(bool))
        y = mm.member.to_numpy(int); rm = mm.is_rare.to_numpy(bool)
        rows.append(dict(channel='mia', method=method, score=r.score,
            tpr_all=r.breakdown['tpr_at_low_fpr'], tpr_rare=r.breakdown['tpr_rare'],
            n_member=int(y.sum()), n_control=int((1-y).sum()),
            n_member_rare=int(y[rm].sum()), n_control_rare=int((1-y[rm]).sum())))
    for method in AIA_METHODS:
        pred = a[method].to_numpy()
        r = score_aia(None, pred[member], aa.onset.to_numpy()[member], aa.time.to_numpy()[member],
            None, pred[~member], aa.onset.to_numpy()[~member], aa.time.to_numpy()[~member], cfg,
            member_rare_mask=rare[member], control_rare_mask=rare[~member])
        row = dict(channel='aia', method=method, score=r.score,
            **{k: r.breakdown[k] for k in ['r_time_all', 'r_time_rare', 'r_time_rare_raw',
                'succ_time_member', 'succ_time_control', 'succ_time_member_rare', 'succ_time_control_rare']},
            n_member=int(member.sum()), n_control=int((~member).sum()),
            n_member_rare=int((rare & member).sum()), n_control_rare=int((rare & ~member).sum()))
        for phi in [0., .05, .1, .2]: row[f'floor_sensitivity_{phi:g}'] = max(0., r.score-phi)
        rows.append(row)
    for row in rows: row.update(target=target, group=group, seed=seed, mechanism=mechanism,
        version=VERSION, official_final_score=False, aia_floor_known=False)
    return pd.DataFrame(rows)


def predict_save(folder, c, mq, aq, bg, cfg, detector):
    folder.mkdir(parents=True, exist_ok=True)
    m = predict_mia(c[COLUMNS], mq[['record_id']+STATIC], bg[COLUMNS], cfg, detector)
    a = predict_aia(c[COLUMNS], aq[['challenge_row_id']+QI], bg[COLUMNS], cfg)
    m.to_csv(folder/'mia.csv', index=False); a.to_csv(folder/'aia.csv', index=False)
    # Read saved output before scoring to make replay independent of transient arrays.
    return read(folder/'mia.csv'), read(folder/'aia.csv')


def checkpoint(folder, metrics):
    metrics.to_csv(folder/'metrics.csv', index=False)
    dump(folder/'checkpoint.json', {'files': {p.name: sha(p) for p in sorted(folder.glob('*.csv'))}})


def load_checkpoint(folder):
    marker = folder/'checkpoint.json'
    if not marker.exists(): return None
    for name, digest in json.loads(marker.read_text())['files'].items():
        assert sha(folder/name) == digest, f'Checkpoint changed: {folder/name}'
    return read(folder/'metrics.csv')


def prelim():
    cfg = config(); own = read(ROOT/'_private_team10/B_self.csv')
    bg = read(ROOT/'_private_team10/A_bg.csv')[COLUMNS]
    detector = json.loads((ROOT/'_private_team10/schema.json').read_text())['rare_detector']
    targets = sorted(int(p.stem.split('_')[-1]) for p in PUBLIC.glob('C_*.csv') if p.stem != 'C_10')
    results = []
    for k in targets:
        folder = OUT/'prelim'/f'target{k}'; saved = load_checkpoint(folder)
        if saved is None:
            c = read(PUBLIC/f'C_{k}.csv'); aq = read(PUBLIC/f'aia_challenge_{k}.csv')
            m, a = predict_save(folder, c, own, aq, bg, cfg, detector)
            pool = read(TRUTH/'pool_ids_10.csv').set_index('record_id').loc[m.record_id]
            assert set(m.record_id) == set(own.record_id)
            mt = pd.DataFrame({'record_id':m.record_id, 'member':pool.pool_id.isin(
                read(TRUTH/f'pool_ids_{k}.csv').pool_id).to_numpy(int), 'is_rare':pool.is_rare.to_numpy()})
            at = read(TRUTH/f'aia_truth_{k}.csv')
            saved = score_predictions(m, a, mt, at, cfg, k, 'prelim_retrospective')
            checkpoint(folder, saved)
        results.append(saved)
        print(f'prelim target{k}: 13 method scores saved', flush=True)
    metrics = pd.concat(results, ignore_index=True); metrics.to_csv(OUT/'prelim_metrics.csv', index=False)
    previous = read(ROOT/'research/runs/20260924-G03c-001/same_challenge_aia.csv')
    joined = metrics[metrics.channel.eq('aia')].merge(previous[['target','method','score']],
        on=['target','method'], suffixes=('_new','_old'), validate='one_to_one')
    error = float(abs(joined.score_new-joined.score_old).max()); assert error < 1e-12
    dump(OUT/'prelim_replay.json', {'pairs':len(joined), 'max_score_error':error,
        'inference_uses_truth':False, 'self10_excluded':True, 'public_truth_is_exploratory':True})


def shadow(seed):
    cfg = config(); own = read(ROOT/'_private_team10/B_self.csv')
    bg = read(ROOT/'_private_team10/A_bg.csv')
    ix = np.random.default_rng(2026100600).permutation(len(bg))
    gix = ix[:1000] if seed == 2026100601 else ix[1000:2000]
    rix = ix[2000:]; assert not set(gix)&set(rix)
    g, r = bg.iloc[gix].reset_index(drop=True), bg.iloc[rix].reset_index(drop=True)
    detector = json.loads((ROOT/'_private_team10/schema.json').read_text())['rare_detector']
    world = OUT/'shadow'/f'seed{seed}'; world.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'bg_position':ix, 'role':['exploration_generation']*1000+
        ['confirmation_generation']*1000+['reference']*len(rix)}).to_csv(world/'partition.csv', index=False)
    rare_ix = np.flatnonzero(own.is_rare.to_numpy(bool)); common_ix = np.flatnonzero(~own.is_rare.to_numpy(bool))
    scale = r[NUM].std(ddof=0).to_numpy(); results = []
    for k in range(3):
        rng = np.random.default_rng(seed + 100*k)
        mix = np.r_[rng.choice(rare_ix, 8, replace=False), rng.choice(common_ix, 92, replace=False)]
        control = np.r_[rng.choice(np.setdiff1d(rare_ix,mix),8,replace=False),
                        rng.choice(np.setdiff1d(common_ix,mix),92,replace=False)]
        base = pd.concat([own.iloc[mix][COLUMNS],g.iloc[rng.choice(len(g),949,replace=False)][COLUMNS]], ignore_index=True)
        noise = base.copy(); noise[NUM] = np.maximum(0, noise[NUM].to_numpy()+rng.normal(size=(1049,7))*scale*.1)
        cluster = base.astype({col:float for col in NUM})
        for positions in base.groupby(['sex','prefecture'],sort=True).indices.values():
            ordered = rng.permutation(positions)
            for start in range(0,len(ordered),5):
                group = ordered[start:start+5]; cluster.loc[group,NUM] = base.loc[group,NUM].mean().to_numpy()
        pure = g.iloc[rng.choice(len(g),1049,replace=True)][COLUMNS].reset_index(drop=True)
        frames = dict(raw=base, noise=noise, cluster=cluster, pure_bg=pure)
        membership = own.record_id.isin(own.iloc[mix].record_id).to_numpy(int)
        mt = own[['record_id','is_rare']].assign(member=membership)
        at = own.iloc[np.r_[mix,control]][['time','onset','is_rare']].reset_index(drop=True)
        at.insert(0,'challenge_row_id',np.arange(len(at))); at['label']=['member']*100+['control']*100
        aq = own.iloc[np.r_[mix,control]][QI].reset_index(drop=True)
        aq.insert(0,'challenge_row_id',np.arange(len(aq)))
        for mechanism, c in frames.items():
            folder = world/f'{mechanism}_{k}'; saved = load_checkpoint(folder)
            if saved is None:
                folder.mkdir(parents=True,exist_ok=True)
                c.to_csv(folder/'C.csv',index=False); aq.to_csv(folder/'query.csv',index=False)
                mt.to_csv(folder/'mia_truth.csv',index=False); at.to_csv(folder/'aia_truth.csv',index=False)
                m, a = predict_save(folder, c, own, aq, r, cfg, detector)
                saved = score_predictions(m, a, mt, at, cfg, k, 'shadow', seed, mechanism)
                checkpoint(folder, saved)
            results.append(saved)
            print(f'shadow {seed}/{mechanism}/{k}: checkpoint',flush=True)
    pd.concat(results,ignore_index=True).to_csv(world/'metrics.csv',index=False)


def paired_summary(metrics, keys):
    rows = []
    for channel, baseline in [('mia','kde_ratio'),('aia','official_1nn')]:
        f = metrics[metrics.channel.eq(channel)]
        base = f[f.method.eq(baseline)].set_index(keys).score
        for method, values in f.groupby('method'):
            delta = values.set_index(keys).score-base
            rows.append(dict(channel=channel,method=method,mean_score=float(values.score.mean()),
                mean_delta=float(delta.mean()), wins=int((delta>1e-12).sum()), losses=int((delta< -1e-12).sum()),
                largest_improvement=float(delta.max()), mean_delta_without_best=float(delta.drop(delta.idxmax()).mean())))
    return pd.DataFrame(rows)


def freeze():
    f = read(OUT/'shadow/seed2026100601/metrics.csv')
    summary = paired_summary(f[f.mechanism.isin(['noise','cluster'])],['target','mechanism'])
    selected = []
    for row in summary.to_dict('records'):
        channel,method = row['channel'],row['method']; values = f[f.method.eq(method)]
        raw,pure = values[values.mechanism.eq('raw')],values[values.mechanism.eq('pure_bg')]
        positive = raw.tpr_all.min()>=.95 if channel=='mia' else raw.score.min()>=.9
        negative = pure.tpr_all.mean()<=.03 and pure.tpr_rare.mean()<=.1 if channel=='mia' else pure.score.mean()<=.1
        row.update(positive_guard=bool(positive),negative_guard=bool(negative),
            progressed=bool(row['mean_delta']>.005 and row['mean_delta_without_best']>0 and positive and negative))
        selected.append(row)
    dump(OUT/'frozen_selection.json',{'exploration_seed':2026100601,'methods':selected,
        'confirmation_seeds':[2026100611,2026100612], 'adjustment_after_confirmation':False})


def fingerprints():
    base = ROOT/'research/runs/20261006-G15b-001'
    manifest = json.loads((ROOT/'_private_team14/main_attack_20260928/IMPORT_MANIFEST.json').read_text())
    dist = ROOT/manifest['files_root']; bg = read(ROOT/'_private_team14/A_bg.csv')[COLUMNS]
    detector = json.loads((ROOT/'_private_team14/schema.json').read_text())['rare_detector']
    features, agreement = [], []
    for k in manifest['targets_including_self']:
        k=int(k)
        if k==14: continue
        c = read(dist/f'C_{k}.csv'); q = read(dist/f'aia_challenge_{k}.csv')
        marker = json.loads((base/f'predictions/target_{k}.json').read_text())
        assert marker['mia_sha256']==sha(base/f'predictions/mia_{k}.csv')
        assert marker['aia_sha256']==sha(base/f'predictions/aia_{k}.csv')
        m,a = read(base/f'predictions/mia_{k}.csv'),read(base/f'predictions/aia_{k}.csv')
        own_ids=read(ROOT/'_private_team14/B.csv').record_id
        keyed_values(m,'record_id',own_ids,'kde_ratio');keyed_values(a,'challenge_row_id',q.challenge_row_id,'official_1nn')
        xq,xc=log_transform(q),log_transform(c);scale=np.maximum(log_transform(bg).std(axis=0,ddof=1),1e-12)
        distances=cdist(xq/scale,xc/scale)
        nearest=[];variance=[]
        for i,row in enumerate(q[QI].itertuples(index=False)):
            ix=block(row,c);order=np.argsort(distances[i,ix],kind='stable')[:20]
            nearest.append(distances[i,ix].min());variance.append(c.time.iloc[ix[order]].var(ddof=0))
        features.append(dict(target=k,**observable_features(c,bg,detector),qi_nn_distance_median=np.median(nearest),
            qi_nn_distance_p95=np.quantile(nearest,.95),nn20_time_variance_mean=np.mean(variance)))
        for method in AIA_METHODS:
            agreement.append(dict(target=k,channel='aia',method=method,agreement=float((a[method]==a.official_1nn).mean()),
                within_half_year=float((abs(a[method]-a.official_1nn)<=.5).mean())))
        for method in MIA_METHODS:
            left=set(m.nlargest(11,method).record_id);right=set(m.nlargest(11,'kde_ratio').record_id)
            agreement.append(dict(target=k,channel='mia',method=method,top11_overlap=len(left&right)/11))
    pd.DataFrame(features).to_csv(OUT/'final_observable_fingerprints.csv',index=False)
    pd.DataFrame(agreement).to_csv(OUT/'final_prediction_agreement.csv',index=False)


def summarize():
    prelim_metrics = read(OUT/'prelim_metrics.csv')
    prelim_summary = paired_summary(prelim_metrics,['target'])
    prelim_summary.to_csv(OUT/'prelim_comparison.csv',index=False)
    f = pd.concat([read(OUT/f'shadow/seed{s}/metrics.csv') for s in [2026100601,2026100611,2026100612]],ignore_index=True)
    f.to_csv(OUT/'shadow_metrics.csv',index=False)
    summary = pd.concat([paired_summary(g[g.mechanism.isin(['noise','cluster'])],['target','mechanism']).assign(seed=s)
                         for s,g in f.groupby('seed')],ignore_index=True)
    summary.to_csv(OUT/'shadow_comparison.csv',index=False)
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); decisions=[]
    for row in frozen['methods']:
        confirmation=summary[(summary.method==row['method'])&summary.seed.ne(2026100601)]
        row['confirmation_reproduced']=bool((confirmation.mean_delta>0).all() and (confirmation.mean_delta_without_best>0).all())
        row['accepted_for_candidate_comparison']=row['progressed'] and row['confirmation_reproduced']
        decisions.append(row)
    dump(OUT/'decision.json',{'methods':decisions,'official_final_score':None,
        'independent_real_cohort':False,'confirmation_not_retuned':True,'baseline_unchanged':True})
    print(prelim_summary.to_string(index=False),flush=True)
    print(summary.to_string(index=False),flush=True)


def replay():
    cfg=config();errors=[];count=0
    for marker in sorted(OUT.glob('**/checkpoint.json')):
        folder=marker.parent;saved=load_checkpoint(folder)
        m,a=read(folder/'mia.csv'),read(folder/'aia.csv')
        if folder.parent.name=='prelim':
            k=int(folder.name.removeprefix('target'))
            pool=read(TRUTH/'pool_ids_10.csv').set_index('record_id').loc[m.record_id]
            mt=pd.DataFrame({'record_id':m.record_id,'member':pool.pool_id.isin(read(TRUTH/f'pool_ids_{k}.csv').pool_id).to_numpy(int),
                'is_rare':pool.is_rare.to_numpy()});at=read(TRUTH/f'aia_truth_{k}.csv')
        else:mt,at=read(folder/'mia_truth.csv'),read(folder/'aia_truth.csv')
        scored=score_predictions(m,a,mt,at,cfg,0,'replay')
        for col in ['score','tpr_all','tpr_rare','r_time_all','r_time_rare']:
            errors.append(float((saved[col]-scored[col]).abs().max()) if (saved[col]-scored[col]).notna().any() else 0.)
        count+=len(saved)
    assert max(errors)<1e-12
    dump(OUT/'replay.json',{'scores_replayed':count,'max_error':max(errors),'checkpoint_hashes_ok':True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prelim','shadow','freeze','fingerprints','summary','replay'],required=True)
    ap.add_argument('--seed',type=int,choices=[2026100601,2026100611,2026100612]);args=ap.parse_args()
    for name,digest in json.loads((OUT/'plan.json').read_text())['inputs'].items():
        assert sha(ROOT/name)==digest, 'Input changed: '+name
    if args.stage=='shadow' and args.seed!=2026100601: assert (OUT/'frozen_selection.json').exists()
    started=time.monotonic()
    if args.stage=='shadow':
        if args.seed is None:ap.error('--seed required for shadow')
        shadow(args.seed)
    else: {'prelim':prelim,'freeze':freeze,'fingerprints':fingerprints,'summary':summarize,'replay':replay}[args.stage]()
    dump(OUT/f'execution_{args.stage}_{args.seed}.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'script_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started,'stage':args.stage,'seed':args.seed})


if __name__=='__main__': main()
