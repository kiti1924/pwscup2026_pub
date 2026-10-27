"""FA112 independent Python ordering/ROC audit of frozen CSV predictions."""
from pathlib import Path
import argparse, json, math, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'research')]
import numpy as np
import pandas as pd
from run_rare_fixed_calibration import load, read, dump, sha, hashes, P38, P98


def tpr(scores, labels, fpr=.01):
    pos=sum(bool(y) for y in labels); neg=len(labels)-pos
    if not pos or not neg:return 0.
    groups={}
    for v,y in zip(scores,labels):
        group=groups.setdefault(float(v),[0,0]);group[0 if y else 1]+=1
    xx=[0.];yy=[0.];tp=fp=0
    for v in sorted(groups,reverse=True):
        p,n=groups[v];tp+=p;fp+=n;xx.append(fp/neg);yy.append(tp/pos)
    return float(np.interp(fpr,xx,yy))


def measure(v,y,r):
    av=tpr(v,y);rv=tpr(v[r],y[r]) if any(y[r]) else float('nan')
    return dict(score=max(av,rv) if math.isfinite(rv) else av,all_tpr=av,rare_tpr=rv)


def independent_order(b,c,r,x):
    eligible=[i for i in range(len(b)) if not r[i] and not x[i]]
    groups={}
    for i in eligible:groups.setdefault((float(c[i]),float(b[i])),[]).append(i)
    slots=sorted((float(b[i]) for i in eligible),reverse=True);out=list(map(float,b));offset=0
    for key in sorted(groups,reverse=True):
        rows=groups[key];values=slots[offset:offset+len(rows)];offset+=len(rows)
        if len(set(values))>1:return np.array(b),True
        for i in rows:out[i]=values[0]
    base_groups={}
    for old,new in zip(b,out):base_groups.setdefault(float(old),set()).add(new)
    if any(len(vals)>1 for vals in base_groups.values()):return np.array(b),True
    return np.array(out),False


def verify(out):
    hashes(out,'input_hashes.json',ROOT);hashes(out,'repair_input_hashes.json',ROOT);hashes(out,'code_hashes.json',ROOT);hashes(out,'prediction_freeze.json',out)
    a=load(out/'inputs.npz');pred=load(out/'predictions.npz');null=load(out/'null_priorities.npz');truth=load(P38/'scoring_truth.npz')
    for k in ['record_id','targets','rare']:np.testing.assert_array_equal(a[k],truth[k])
    ids,targets,r,x=(a[k] for k in ['record_id','targets','rare','excluded']);y=truth['membership'].astype(bool)
    source=load(P98/'predictions.npz');np.testing.assert_array_equal(pred['s3'],source['s3_baseline']);np.testing.assert_array_equal(pred['fa98'],source['rare_slot_raw_peer_hybrid'])
    csvs={};order_checks=0;metric_checks=0
    for name,v in pred.items():
        csv=read(out/(name+'.csv')).set_index('record_id');assert csv.index.is_unique and set(csv.index)==set(ids)
        vals=csv.loc[ids,list(map(str,targets))].to_numpy();np.testing.assert_array_equal(vals,v);csvs[name]=vals
        if name in ['s3','fa98']:continue
        priority=(a['calibration'] if name=='conditional' else a['global_control'] if name=='global_control' else pred['s3'] if name=='identity' else null[name])
        np.testing.assert_array_equal(vals[r|x].view(np.uint64),pred['s3'][r|x].view(np.uint64))
        np.testing.assert_array_equal(np.sort(vals,axis=0),np.sort(pred['s3'],axis=0))
        for j,k in enumerate(targets):
            expect,fallback=independent_order(pred['s3'][:,j],priority[:,j],r,x)
            np.testing.assert_array_equal(vals[:,j],expect);order_checks+=1
    scores=read(out/'scores.csv');assert len(scores)==len(pred)*len(targets) and not scores[['method','target']].duplicated().any()
    for row in scores.itertuples():
        j=list(targets).index(row.target);m=measure(csvs[row.method][:,j],y[:,j],r)
        for key,value in m.items():
            assert np.isclose(value,getattr(row,key),rtol=0,atol=1e-12,equal_nan=True);metric_checks+=1
        assert row.members==sum(y[:,j]) and row.controls==sum(~y[:,j])
        assert row.rare_members==sum(y[r,j]) and row.rare_controls==sum(~y[r,j])
    deletion=read(out/'rare_one_person_deletion.csv')
    assert len(deletion)==int(r.sum())*len(targets)
    for row in deletion.itertuples():
        keep=ids!=row.removed_record_id;j=list(targets).index(row.target)
        for name in ['s3','conditional','fa98']:
            expected=measure(csvs[name][keep,j],y[keep,j],r[keep])['score']
            assert np.isclose(expected,getattr(row,name),rtol=0,atol=1e-12);metric_checks+=1
    plan=json.loads((out/'plan.json').read_text());summary=json.loads((out/'summary.json').read_text());aggregate_checks=0
    for row in summary:
        ks=plan['targets_common19'] if row['scope']=='common19' else list(targets)
        part=scores[scores.target.isin(ks)];b=part[part.method.eq('s3')].set_index('target');s=part[part.method.eq('fa98')].set_index('target');c=part[part.method.eq(row['method'])].set_index('target');g=c.score-b.score
        computed=dict(score=c.score.mean(),all_tpr=c.all_tpr.mean(),rare_tpr=c.rare_tpr.mean(),gain=g.mean(),gain_vs_fa98=(c.score-s.score).mean(),drop_largest=(sum(g)-max(g))/(len(g)-1))
        for key,value in computed.items():assert np.isclose(value,row[key],atol=1e-12,rtol=0);aggregate_checks+=1
        assert row['positive_targets']==list(map(int,g.index[g>1e-12])) and row['negative_targets']==list(map(int,g.index[g< -1e-12]))
        if row['method']=='conditional':
            d=deletion[deletion.target.isin(ks)];worst=float((d.conditional-d.s3).groupby(d.removed_record_id).mean().min())
            nullg=[float((part[part.method.eq(n)].set_index('target').score-b.score).mean()) for n in pred if n.startswith('null_')]
            q95=float(np.quantile(nullg,.95));assert np.isclose(worst,row['worst_rare_deletion_gain'],atol=1e-12,rtol=0)
            assert np.isclose(q95,row['null_gain_q95'],atol=1e-12,rtol=0)
            assert row['screen_pass']==bool(g.mean()>=.005 and computed['drop_largest']>0 and worst>=0 and g.mean()>q95)
    fixtures=load(out/'fixtures.npz')
    for b,c,r0,expected in [(fixtures['tied_baseline'],fixtures['tied_calibration'],np.zeros(4,bool),fixtures['tied_result']),
                           (fixtures['tied_baseline'],fixtures['protected_tie_calibration'],fixtures['protected_tie_mask'],fixtures['protected_tie_result'])]:
        v,f=independent_order(b[:,0],c[:,0],r0,np.zeros(4,bool));assert f;np.testing.assert_array_equal(v,expected[:,0])
    from rare_fixed_calibration import reassign
    perm=np.random.default_rng(plan['seed']+101).permutation(len(ids));inv=np.argsort(perm)
    new,_=reassign(pred['s3'][perm],a['calibration'][perm],r[perm],x[perm]);np.testing.assert_array_equal(new[inv],pred['conditional'])
    hashes(out,'prediction_freeze.json',out)
    dump(out/'verification.json',dict(passed=True,independent_order_columns=order_checks,scalar_metric_checks=metric_checks,
         aggregate_checks=aggregate_checks,all_CSV_bit_exact=True,protected_bit_exact=True,baseline_tie_groups_preserved=True,
         shuffle_pass=True,verifier_hash=sha(Path(__file__))))
    print(json.dumps(json.loads((out/'verification.json').read_text())))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
