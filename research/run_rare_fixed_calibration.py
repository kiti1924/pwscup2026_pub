"""FA112 separate public inference/freeze, scoring, and independent verification."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'research'), str(ROOT/'pwscup2026-kit/starter/src')]
import numpy as np
import pandas as pd
import yaml
from rare_fixed_calibration import priority, reassign
R = ROOT/'research/runs'
P06 = R/'20261010-CL06-conditional-null-mia-001'
P36 = R/'20261010-FA36-joint-membership-001'
P38 = R/'20261010-FA38-exclusive-ranking-001'
P98 = R/'20261010-FA98-mia-rare-slots-001'


def load(p):
    with np.load(p, allow_pickle=False) as z: return dict(z)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
def read(p): return pd.read_csv(p, float_precision='round_trip')
def hashes(out, name, base):
    for p, h in json.loads((out/name).read_text()).items(): assert sha(base/p) == h, p


def infer(out, plan):
    assert not (out/'predictions.npz').exists(), 'Never overwrite frozen predictions'
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            s = str(args[0])
            if any(v in s for v in ['scoring_truth', '/sources/truth', '/sources/attacks', 'parsed_details']):
                raise RuntimeError('Truth access during inference')
    # Hashing pinned truth bytes is performed by main before this guard; inference
    # never loads their contents. All membership arrays remain scoring-only.
    sys.addaudithook(guard)
    a = load(P06/'public_arrays.npz'); b = load(P98/'inputs.npz')
    p = load(P98/'predictions.npz'); f = load(P36/'inference_arrays.npz')
    for key in ['record_id', 'targets', 'rare', 'excluded']: np.testing.assert_array_equal(a[key], b[key])
    for key in ['record_id', 'targets', 'rare']: np.testing.assert_array_equal(a[key], f[key])
    ids, targets, rare, excluded = (a[k] for k in ['record_id', 'targets', 'rare', 'excluded'])
    assert len(ids) == len(np.unique(ids)) == 1049 and len(targets) == 20
    own = pd.read_csv(ROOT/'_private_team10/B_self.csv', usecols=['record_id','is_rare'], float_precision='round_trip').set_index('record_id').loc[ids]
    np.testing.assert_array_equal(own.is_rare.to_numpy(bool), rare)
    base = p['s3_baseline']; strong = p['rare_slot_raw_peer_hybrid']
    cal = priority(a['e0__blend'], f['home'])
    ctrl = priority(a['e0__ctrl_all'], f['home'])
    new, reports = reassign(base, cal, rare, excluded)
    identity, identity_report = reassign(base, base, rare, excluded)
    global_control, global_report = reassign(base, ctrl, rare, excluded)
    np.testing.assert_array_equal(identity, base)
    np.testing.assert_array_equal(global_control, base)
    pred = dict(s3=base, fa98=strong, conditional=new, identity=identity, global_control=global_control)
    null_priorities = {}; null_reports = {}
    for seed in range(19):
        rng = np.random.default_rng(plan['seed']+seed)
        nc = cal.copy()
        for j in range(len(targets)):
            for h in [False, True]:
                ix = np.flatnonzero(~rare & ~excluded & (f['home'] == h))
                if np.unique(base[ix, j]).size != len(ix):
                    nc[:, j] = base[:, j]; break
                order = ix[np.argsort(base[ix, j])]
                nc[order, j] = nc[order[rng.permutation(len(order))], j]
        name = f'null_{seed:02d}'
        pred[name], null_reports[name] = reassign(base, nc, rare, excluded)
        null_priorities[name] = nc
    perm = np.random.default_rng(plan['seed']+100).permutation(len(ids)); inv = np.argsort(perm)
    shuffled, _ = reassign(base[perm], cal[perm], rare[perm], excluded[perm])
    np.testing.assert_array_equal(shuffled[inv], new)
    # Minimal algorithm fixtures: correct priority recovers reversed slots; ties
    # retaining baseline order pass, incompatible baseline ties trigger fallback.
    rb = np.array([[.1], [.2], [.3], [.4]])
    positive, _ = reassign(rb, -rb, np.zeros(4, bool), np.zeros(4, bool))
    np.testing.assert_array_equal(positive[:, 0], [.4,.3,.2,.1])
    tied_priority, _ = reassign(rb, np.ones_like(rb), np.zeros(4, bool), np.zeros(4, bool))
    np.testing.assert_array_equal(tied_priority, rb)
    tb = np.array([[.1],[.1],[.2],[.3]])
    tc = np.array([[4.],[1.],[3.],[2.]])
    unchanged, tr = reassign(tb, tc, np.zeros(4, bool), np.zeros(4, bool))
    np.testing.assert_array_equal(unchanged, tb); assert tr[0]['fallback'] is not None
    protected = np.array([True,False,False,False])
    protected_cal = np.array([[1.],[4.],[3.],[2.]])
    unchanged2, tr2 = reassign(tb, protected_cal, protected, np.zeros(4, bool))
    np.testing.assert_array_equal(unchanged2, tb); assert tr2[0]['fallback'] is not None
    np.savez_compressed(out/'fixtures.npz', baseline=rb, positive=positive, tied_priority=tied_priority,
                        tied_baseline=tb, tied_calibration=tc, tied_result=unchanged,
                        protected_tie_mask=protected, protected_tie_calibration=protected_cal, protected_tie_result=unchanged2)
    np.savez_compressed(out/'inputs.npz', record_id=ids, targets=targets, rare=rare, excluded=excluded,
                        home=f['home'], calibration=cal, global_control=ctrl)
    np.savez_compressed(out/'null_priorities.npz', **null_priorities)
    np.savez_compressed(out/'predictions.npz', **pred)
    for name, v in pred.items():
        df = pd.DataFrame(v, columns=list(map(str, targets))); df.insert(0, 'record_id', ids)
        df.to_csv(out/(name+'.csv'), index=False, lineterminator='\n')
        np.testing.assert_array_equal(read(out/(name+'.csv')).iloc[:, 1:].to_numpy(), v)
    dump(out/'tie_reports.json', dict(conditional=reports, identity=identity_report, global_control=global_report, null=null_reports))
    dump(out/'prediction_freeze.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.csv','.npz'] or p.name == 'tie_reports.json'})
    dump(out/'inference_complete.json', dict(truth_read=False, prediction_matrices=len(pred), rows=len(ids), targets=len(targets),
         protected_rare=int(rare.sum()), protected_excluded=int(excluded.sum()), eligible=int((~rare & ~excluded).sum()),
         shuffle_invariant=True, fixtures_pass=True, algorithm_control_not_population_evidence=True))


def metric(v, y, rare, cfg):
    from pwscup2026.scoring.attack_score import score_mia
    s = score_mia(v, y, cfg, rare_mask=rare)
    return dict(score=s.score, all_tpr=s.breakdown['tpr_at_low_fpr'], rare_tpr=s.breakdown['tpr_rare'])


def score(out, plan):
    assert not (out/'scores.csv').exists(), 'No overwrite'
    hashes(out, 'prediction_freeze.json', out)
    # Require that the prediction freeze is already committed and on the shared branch.
    rel = str((out/'prediction_freeze.json').relative_to(ROOT))
    commit = subprocess.check_output(['git','log','-1','--format=%H','--',rel], cwd=ROOT, text=True).strip()
    assert commit
    subprocess.run(['git','merge-base','--is-ancestor',commit,'@{u}'], cwd=ROOT, check=True)
    a = load(out/'inputs.npz'); pred = load(out/'predictions.npz'); t = load(P38/'scoring_truth.npz')
    for key in ['record_id','targets','rare']: np.testing.assert_array_equal(a[key], t[key])
    cfg = yaml.safe_load((ROOT/'_private_team10/scoring_config.yaml').read_text())
    rare, y, targets = a['rare'], t['membership'].astype(bool), a['targets']; rows=[]
    for name, v in pred.items():
        for j, k in enumerate(targets):
            m = metric(v[:,j], y[:,j], rare, cfg)
            assert np.isfinite(m['score']) and np.isfinite(m['all_tpr'])
            rows.append(dict(method=name,target=int(k),members=int(y[:,j].sum()),controls=int((~y[:,j]).sum()),
                             rare_members=int(y[rare,j].sum()),rare_controls=int((~y[rare,j]).sum()),**m))
    frame = pd.DataFrame(rows); frame.to_csv(out/'scores.csv', index=False)
    baseline = frame[frame.method.eq('s3')].set_index('target')
    for name in pred:
        if name != 'fa98':
            np.testing.assert_allclose(frame[frame.method.eq(name)].set_index('target').rare_tpr,
                                       baseline.rare_tpr, rtol=0, atol=0, equal_nan=True)
    deletion=[]
    for ix in np.flatnonzero(rare):
        keep = np.arange(len(rare)) != ix
        for j,k in enumerate(targets):
            vals = {n:metric(pred[n][keep,j],y[keep,j],rare[keep],cfg)['score'] for n in ['s3','conditional','fa98']}
            deletion.append(dict(removed_record_id=int(a['record_id'][ix]),target=int(k),**vals))
    d = pd.DataFrame(deletion); d.to_csv(out/'rare_one_person_deletion.csv', index=False)
    summaries=[]
    for scope,ks in [('common19',plan['targets_common19']),('aux20',targets.tolist())]:
        subset=frame[frame.target.isin(ks)]; b=subset[subset.method.eq('s3')].set_index('target')
        strong=subset[subset.method.eq('fa98')].set_index('target')
        nullg=[]
        for name in pred:
            q=subset[subset.method.eq(name)].set_index('target'); gain=q.score-b.score
            row=dict(scope=scope,method=name,score=float(q.score.mean()),all_tpr=float(q.all_tpr.mean()),rare_tpr=float(q.rare_tpr.mean()),
                     gain=float(gain.mean()),gain_vs_fa98=float((q.score-strong.score).mean()),drop_largest=float((gain.sum()-gain.max())/(len(gain)-1)),
                     positive_targets=[int(k) for k in gain.index[gain>1e-12]],negative_targets=[int(k) for k in gain.index[gain< -1e-12]])
            summaries.append(row)
            if name.startswith('null_'):nullg.append(row['gain'])
        cand=next(v for v in summaries if v['scope']==scope and v['method']=='conditional')
        ds=d[d.target.isin(ks)]; wg=float((ds.conditional-ds.s3).groupby(ds.removed_record_id).mean().min())
        cand.update(worst_rare_deletion_gain=wg,null_gain_q95=float(np.quantile(nullg,.95)))
        cand['screen_pass']=bool(cand['gain']>=.005 and cand['drop_largest']>0 and wg>=0 and cand['gain']>cand['null_gain_q95'])
    dump(out/'summary.json',summaries)
    dump(out/'score_complete.json',dict(frozen_commit=commit,rows=len(frame),rare_values_identical=True,main_scored=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['infer','score'],required=True);args=ap.parse_args()
    out=args.out.resolve();plan=json.loads((out/'plan.json').read_text())
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(plan['deadline_jst'])
    hashes(out,'input_hashes.json',ROOT)
    hashes(out,'repair_input_hashes.json',ROOT)
    if args.stage=='infer':
        dump(out/'code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/rare_fixed_calibration.py']})
    else: hashes(out,'code_hashes.json',ROOT)
    start=time.monotonic();globals()[args.stage](out,plan)
    elapsed=time.monotonic()-start;assert elapsed<plan['compute_budget_seconds']
    dump(out/(args.stage+'_timing.json'),dict(seconds=elapsed))

if __name__=='__main__':main()
