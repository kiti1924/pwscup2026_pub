"""FA101 main public construction only. No scorer/teacher/truth imports."""
from pathlib import Path
import argparse, hashlib, io, json, subprocess, sys, time, zipfile
import numpy as np
import pandas as pd
import yaml
import nearest_decoy as nd
from main_attack_contrasts import raw_peer, age_mask, STATIC, QI
from mia_rare_slots import reorder
from known_private_exclusion import match

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/runs'; P95=R/'20261010-FA95-fa55-public-api-001'; P100=R/'20261010-FA100-age-proxy-001/execution/run-001'
PUBLIC=ROOT/'_private_team14/main_attack_20260928/files'
NAMES=['FA97_gated_rank2','FA97_ungated_rank2','FA100_age5','FA100_age10']
def read(p,**kwargs):return pd.read_csv(p,float_precision='round_trip',**kwargs)
def load(p):
    with np.load(p) as f:return dict(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csv(p,v):pd.DataFrame(v).to_csv(p,index=False,lineterminator='\n')


def run(out):
    started=time.monotonic();plan=json.loads((out/'plan.json').read_text());ks=plan['targets']
    assert not (out/'complete.json').exists()
    accessed=set()
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            text=str(args[0])
            if '/sources/truth' in text or '/sources/attacks' in text or 'scoring_truth' in text:
                raise RuntimeError('Main construction tried truth/teacher input')
            if text.startswith(str(ROOT)):accessed.add(text[len(str(ROOT))+1:])
    sys.addaudithook(hook)
    for p,h in json.loads((out/'input_hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
    # Port check: same historical lawful E1 and masks, no truth reads.
    old=load(R/'20261010-FA36-joint-membership-001/inference_arrays.npz')
    pp=ROOT/'attack_phase/data/public/prelim_attack_20260825'
    bp=read(ROOT/'_private_team10/B_self.csv',usecols=['record_id']+STATIC).set_index('record_id').loc[old['record_id']]
    bgp=read(ROOT/'_private_team10/A_bg.csv',usecols=STATIC)
    cp={int(k):read(pp/f'C_{k}.csv',usecols=STATIC) for k in old['targets']}
    ep,_,_=raw_peer(bp[STATIC],bgp[STATIC],cp)
    np.testing.assert_array_equal(ep,old['e1'])
    for k in old['targets']:
        saved=load(P95/f'preliminary/target{k}/evidence.npz');q=read(P100/f'target{k}/query.csv');m=load(P100/f'target{k}/masks.npz')
        for width in [5,10]:
            mask,_=age_mask(q,saved['mean_evidence'],saved['mask'],width)
            np.testing.assert_array_equal(mask,m['age'+str(width)])
    dump(out/'port_verification.json',dict(preliminary_E1_exact=True,preliminary_age_masks_exact=40,truth_read=False))
    scale=load(P95/'main/scale.npz');scale.update(columns=nd.api.NUM,space='l1_log5',base_space='log5')
    answers={n:[] for n in NAMES};features=[];cells=[];selection=[]
    for k in ks:
        assert time.monotonic()-started<plan['compute_seconds_max']
        folder=out/f'target{k}';folder.mkdir()
        q=read(P95/f'main/target{k}/query.csv');saved=load(P95/f'main/target{k}/evidence.npz')
        rawq=read(PUBLIC/f'aia_challenge_{k}.csv')
        nd.api.check(q,query=True,expected=rawq.challenge_row_id)
        pd.testing.assert_frame_equal(q.set_index('challenge_row_id').sort_index(),rawq.set_index('challenge_row_id').sort_index())
        np.testing.assert_array_equal(q.challenge_row_id,saved['challenge_row_id'])
        result=nd.predict(read(PUBLIC/f'C_{k}.csv'),q,scale,saved['mask'])
        np.testing.assert_array_equal(result['ranked_times'][:,0],saved['donor_time'])
        rank2=result['ranked_times'][:,1];pred=[np.where(saved['mask'],rank2,12.),rank2]
        masks=[saved['mask'],np.ones(len(q),bool)];group_sets=[]
        for width in [5,10]:
            mask,groups=age_mask(q,saved['mean_evidence'],saved['mask'],width)
            masks.append(mask);pred.append(np.where(mask,saved['donor_time'],12.));group_sets.append(groups)
            for group in groups:
                ix=group['indices'];cells.append(dict(target=k,width=width,sex=group['sex'],prefecture=group['prefecture'],age_bin=group['age_bin'],rows=len(ix),odd=bool(len(ix)%2),baseline_selected=int(saved['mask'][ix].sum()),new_selected=int(mask[ix].sum())))
        for name,v,mask in zip(NAMES,pred,masks):
            frame=pd.DataFrame(dict(target_k=k,challenge_row_id=q.challenge_row_id,time_hat=v));answers[name].append(frame)
            features.append(dict(target=k,method=name,rows=len(q),selected=int(mask.sum()),sentinel12=int((v==12).sum()),late_ge9=int((v>=9).sum()),late_selected_ge9=int(((v>=9)&mask).sum()),unique_selected_times=int(len(np.unique(v[mask]))),rank1_rank2_time_disagreement=int((result['ranked_times'][:,0]!=rank2).sum()),rank2_distance_mean=float(result['distances'][:,1].mean()),relative_gap_mean=float(((result['distances'][:,1]-result['distances'][:,0])/np.maximum(result['distances'][:,1],1e-12)).mean())))
            for sex in sorted(q.sex.unique()):
                ix=q.sex.eq(sex).to_numpy();selection.append(dict(target=k,method=name,sex=int(sex),rows=int(ix.sum()),baseline_selected=int(saved['mask'][ix].sum()),new_selected=int(mask[ix].sum())))
        np.savez_compressed(folder/'public_predictions.npz',ids=q.challenge_row_id,ranked_times=result['ranked_times'],distances=result['distances'],predictions=np.asarray(pred),masks=np.asarray(masks),mean_evidence=saved['mean_evidence'])
        dump(folder/'checkpoint.json',dict(sha256=sha(folder/'public_predictions.npz'),truth_read=False))
    aia={}
    for name,frames in answers.items():
        value=pd.concat(frames,ignore_index=True).sort_values(['target_k','challenge_row_id'])
        assert len(value)==7172 and not value.duplicated(['target_k','challenge_row_id']).any()
        aia[name]=value;value.to_csv(out/(name+'_F_aia.csv'),index=False,lineterminator='\n')
    csv(out/'public_characteristics.csv',features);csv(out/'public_age_cells.csv',cells);csv(out/'selection_by_sex.csv',selection)
    # Only own lawful MIA static/rarity is admitted here; outcomes are not parsed.
    own=read(ROOT/'_private_team14/B_self.csv',usecols=['record_id']+STATIC+['is_rare'])
    baseline_path=ROOT/'research/submissions/four-measured-backup-20261010/candidate/F_mia.csv'
    assert sha(baseline_path)=='2910205c4ab3b6e11098201024fd2d5d480cbb11f265fea798567e0d36e47e2c'
    base_df=read(baseline_path);ids=base_df.record_id.to_numpy();own=own.set_index('record_id').loc[ids].reset_index()
    base=base_df[[str(k) for k in ks]].to_numpy();rare=own.is_rare.to_numpy(bool)
    excluded,_,_=match(own[['record_id']+QI],read(PUBLIC/'aia_challenge_14.csv'))
    previous=load(R/'20261010-FA37-known-private-001/main/matches.npz')
    np.testing.assert_array_equal(previous['record_id'],ids);np.testing.assert_array_equal(previous['strict_mask'],excluded)
    bg=read(ROOT/'_private_team14/A_bg.csv',usecols=STATIC);cs={k:read(PUBLIC/f'C_{k}.csv',usecols=STATIC) for k in ks}
    e1,ld,counts=raw_peer(own[STATIC],bg[STATIC],cs)
    home=own.prefecture.eq(own.prefecture.mode().iloc[0]).to_numpy();evidence=e1+np.log(.087)*home[:,None]
    new=reorder(base,evidence,rare,excluded,ids)
    np.testing.assert_array_equal(base[~rare],new[~rare]);np.testing.assert_array_equal(base[excluded],new[excluded]);np.testing.assert_array_equal(np.sort(base[rare],axis=0),np.sort(new[rare],axis=0))
    csv(out/'mia_peer_C_counts.csv',counts)
    np.savez_compressed(out/'mia_public_arrays.npz',record_id=ids,targets=ks,rare=rare,excluded=excluded,home=home,e1=e1,log_distances=ld,baseline=base,hybrid=new)
    frame=pd.DataFrame(new,columns=[str(k) for k in ks]);frame.insert(0,'record_id',ids);frame.to_csv(out/'FA98_diagnostic_F_mia.csv',index=False,lineterminator='\n')
    changes=[];tail=[]
    for j,k in enumerate(ks):
        a=np.lexsort((ids,-base[:,j]));b=np.lexsort((ids,-new[:,j]));ri=np.flatnonzero(rare)
        ra=ri[np.lexsort((ids[ri],-base[ri,j]))];rb=ri[np.lexsort((ids[ri],-new[ri,j]))]
        for ix in np.flatnonzero(base[:,j]!=new[:,j]):changes.append(dict(target=k,record_id=int(ids[ix]),is_rare=bool(rare[ix]),excluded=bool(excluded[ix]),baseline_score=base[ix,j],hybrid_score=new[ix,j]))
        for count in [1,15,25]:
            tail.append(dict(target=k,tail=count,baseline_excluded=int(excluded[a[:count]].sum()),hybrid_excluded=int(excluded[b[:count]].sum()),crossing_in=len(set(ids[b[:count]])-set(ids[a[:count]])),crossing_out=len(set(ids[a[:count]])-set(ids[b[:count]])),rare_top1_before=int(ids[ra[0]]),rare_top1_after=int(ids[rb[0]])))
    csv(out/'mia_changed_rare_identities.csv',changes);csv(out/'mia_top_tail_crossings.csv',tail)
    dump(out/'mia_invariants.json',dict(rare=int(rare.sum()),excluded=int(excluded.sum()),excluded_rare=int((excluded&rare).sum()),eligible_rare=int((rare&~excluded).sum()),nonrare_exact=True,excluded_exact=True,rare_slots_exact=True,changed_cells=len(changes),source972302_bytes_sha256=sha(baseline_path),main_score=None,FA98_screen_still_failed=True))
    # Comparison only, after new inference is persisted; historical values never feed it.
    measured={}
    for sub in [965017,971407,972302,974256]:
        with zipfile.ZipFile(ROOT/f'research/submissions/four-measured-backup-20261010/inputs/{sub}.zip') as z:
            measured[str(sub)]=read(io.BytesIO(z.read('F_aia.csv')))
    all_aia={**measured,**aia};comparisons=[]
    for name,value in aia.items():
        for other,reference in all_aia.items():
            if other==name:continue
            a=value.set_index(['target_k','challenge_row_id']).time_hat;b=reference.set_index(['target_k','challenge_row_id']).time_hat.reindex(a.index)
            assert b.notna().all()
            for k in ks:
                av=a.loc[k].to_numpy();bv=b.loc[k].to_numpy()
                comparisons.append(dict(method=name,reference=other,target=k,answer_differences=int((av!=bv).sum()),non12_mask_differences=int(((av!=12)!=(bv!=12)).sum()),new_time_values=int(len(set(av)-set(bv)))))
    csv(out/'answer_pairwise_differences.csv',comparisons)
    # Format verification once per output, no scoring/ZIP work.
    sys.path.insert(0,str(ROOT/'pwscup2026-kit/starter/src'))
    from pwscup2026.kit.validate import validate_mia_submission,validate_aia_submission
    cfg=yaml.safe_load((ROOT/'_private_team14/scoring_config.yaml').read_text())
    strict=dict(team_id=14,mia_columns=json.loads((PUBLIC/'mia_columns.json').read_text()),attacker_record_ids=set(ids),challenge={k:set(read(PUBLIC/f'aia_challenge_{k}.csv').challenge_row_id) for k in ks})
    validation=[]
    for name,value in aia.items():
        v=validate_aia_submission(value,cfg,strict);assert v.ok,name
        validation.append(dict(artifact=name+'_F_aia.csv',validator='AIA',passed=True))
    v=validate_mia_submission(frame,cfg,strict);assert v.ok,'MIA format validation failed'
    validation.append(dict(artifact='FA98_diagnostic_F_mia.csv',validator='MIA',passed=True))
    dump(out/'validation.json',validation)
    for name in aia:
        np.testing.assert_array_equal(read(out/(name+'_F_aia.csv')).time_hat.to_numpy(),aia[name].time_hat.to_numpy())
    np.testing.assert_array_equal(read(out/'FA98_diagnostic_F_mia.csv')[[str(k) for k in ks]].to_numpy(),new)
    dump(out/'access_audit.json',dict(truth_teacher_reads=0,main_score_reads=0,repository_paths=sorted(accessed)))
    dump(out/'code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/main_attack_contrasts.py',ROOT/'research/nearest_decoy.py',ROOT/'research/mia_rare_slots.py',ROOT/'research/known_private_exclusion.py']})
    dump(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['run.log','artifact_hashes.json']})
    assert time.monotonic()-started<plan['compute_seconds_max']
    dump(out/'complete.json',dict(git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),compute_seconds=time.monotonic()-started,aia_candidates=4,aia_keys_each=7172,mia_rows=1049,official_format_checks=5,models_trained=0,new_scores=0,new_submissions=0,new_ZIPs=0))
    print((out/'complete.json').read_text())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
